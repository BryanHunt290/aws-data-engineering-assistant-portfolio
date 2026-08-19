"""Storage abstractions and an S3 adapter for the knowledge hierarchy."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Protocol, runtime_checkable
from urllib.parse import quote


class ConditionalStorageConflictError(RuntimeError):
    """Raised when an optimistic object write loses a version race."""


@dataclass(frozen=True)
class VersionedJsonObject:
    """A JSON object plus its opaque storage version token."""

    payload: dict[str, Any] | None
    version: str | None


class KnowledgeKeys:
    """Canonical object keys within the existing knowledge bucket."""

    ROOT = "knowledge"
    RAW = f"{ROOT}/raw"
    PROCESSED = f"{ROOT}/processed"
    CHUNKS = f"{ROOT}/chunks"
    EMBEDDINGS = f"{ROOT}/embeddings"
    METADATA = f"{ROOT}/metadata"
    MEDIA = f"{ROOT}/media"
    QUARANTINE = f"{ROOT}/quarantine"
    MANIFEST = f"{METADATA}/manifest.json"

    @classmethod
    def raw(cls, document_id: str, filename: str) -> str:
        return f"{cls.RAW}/{document_id}/{filename}"

    @classmethod
    def processed(cls, document_id: str) -> str:
        return f"{cls.PROCESSED}/{document_id}.txt"

    @classmethod
    def chunks(cls, document_id: str) -> str:
        return f"{cls.CHUNKS}/{document_id}.json"

    @classmethod
    def embeddings(cls, document_id: str) -> str:
        return f"{cls.EMBEDDINGS}/{document_id}.json"

    @classmethod
    def embedding_record(cls, document_id: str, chunk_id: str) -> str:
        encoded_chunk_id = quote(chunk_id, safe="")
        return f"{cls.EMBEDDINGS}/{document_id}/{encoded_chunk_id}.json"

    @classmethod
    def metadata(cls, document_id: str) -> str:
        return f"{cls.METADATA}/{document_id}.json"

    @classmethod
    def storage_only_metadata(cls, object_id: str) -> str:
        return f"{cls.METADATA}/storage-only/{object_id}.json"

    @classmethod
    def media(
        cls,
        client_id: str,
        environment: str,
        media_type: str,
        object_id: str,
        filename: str,
    ) -> str:
        return (
            f"{cls.MEDIA}/{client_id}/{environment}/{media_type}/"
            f"{object_id}/{filename}"
        )

    @classmethod
    def quarantine(
        cls,
        client_id: str,
        environment: str,
        object_id: str,
        filename: str,
    ) -> str:
        return (
            f"{cls.QUARANTINE}/{client_id}/{environment}/"
            f"{object_id}/{filename}"
        )


@runtime_checkable
class KnowledgeStorage(Protocol):
    """Minimal object-store interface used by knowledge ingestion."""

    def put_bytes(
        self,
        key: str,
        content: bytes,
        *,
        content_type: str,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        """Store bytes at a canonical knowledge key."""

    def put_json(self, key: str, payload: Mapping[str, Any]) -> None:
        """Serialize and store a JSON object."""

    def get_json(self, key: str) -> dict[str, Any] | None:
        """Load a JSON object, returning None when it does not exist."""


@runtime_checkable
class ConditionalKnowledgeStorage(KnowledgeStorage, Protocol):
    """Optional optimistic-write extension for aggregate objects."""

    def get_json_versioned(self, key: str) -> VersionedJsonObject:
        """Load JSON and its opaque storage version."""

    def put_json_if_version(
        self,
        key: str,
        payload: Mapping[str, Any],
        expected_version: str | None,
    ) -> None:
        """Write only when the current version matches the expectation."""


class S3KnowledgeStorage:
    """KnowledgeStorage implementation backed by the existing S3 bucket."""

    def __init__(self, bucket_name: str, s3_client: Any) -> None:
        if not bucket_name.strip():
            raise ValueError("bucket_name cannot be empty")
        self._bucket_name = bucket_name
        self._s3_client = s3_client

    @classmethod
    def from_boto3(cls, bucket_name: str) -> "S3KnowledgeStorage":
        """Build the adapter using the default boto3 credential chain."""

        import boto3

        return cls(bucket_name, boto3.client("s3"))

    def put_bytes(
        self,
        key: str,
        content: bytes,
        *,
        content_type: str,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        self._s3_client.put_object(
            Bucket=self._bucket_name,
            Key=key,
            Body=content,
            ContentType=content_type,
            Metadata=dict(metadata or {}),
        )

    def put_json(self, key: str, payload: Mapping[str, Any]) -> None:
        content = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.put_bytes(
            key,
            content,
            content_type="application/json",
        )

    def get_json(self, key: str) -> dict[str, Any] | None:
        try:
            response = self._s3_client.get_object(
                Bucket=self._bucket_name,
                Key=key,
            )
        except Exception as error:
            if self._is_missing_key(error):
                return None
            if self._is_access_denied(error) and not self._object_exists(key):
                # S3 returns 403 rather than 404 for a missing key when the
                # caller lacks unrestricted ListBucket. Use the role's
                # prefix-scoped list permission to distinguish that case
                # without granting visibility into unrelated prefixes.
                return None
            raise

        payload = response["Body"].read()
        parsed = json.loads(payload.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected a JSON object at '{key}'")
        return parsed

    def get_json_versioned(self, key: str) -> VersionedJsonObject:
        try:
            response = self._s3_client.get_object(
                Bucket=self._bucket_name,
                Key=key,
            )
        except Exception as error:
            if self._is_missing_key(error):
                return VersionedJsonObject(None, None)
            if self._is_access_denied(error) and not self._object_exists(key):
                return VersionedJsonObject(None, None)
            raise
        payload = response["Body"].read()
        parsed = json.loads(payload.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected a JSON object at '{key}'")
        etag = response.get("ETag")
        version = etag.strip() if isinstance(etag, str) and etag.strip() else None
        if version is None:
            raise ValueError(f"Expected an ETag for JSON object at '{key}'")
        return VersionedJsonObject(parsed, version)

    def put_json_if_version(
        self,
        key: str,
        payload: Mapping[str, Any],
        expected_version: str | None,
    ) -> None:
        parameters: dict[str, Any] = {
            "Bucket": self._bucket_name,
            "Key": key,
            "Body": json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
            "ContentType": "application/json",
        }
        parameters[
            "IfNoneMatch" if expected_version is None else "IfMatch"
        ] = "*" if expected_version is None else expected_version
        try:
            self._s3_client.put_object(**parameters)
        except Exception as error:
            if self._is_write_conflict(error):
                raise ConditionalStorageConflictError(
                    "Conditional object write conflict"
                ) from error
            raise

    @staticmethod
    def _is_missing_key(error: Exception) -> bool:
        response = getattr(error, "response", {})
        error_details = (
            response.get("Error", {}) if isinstance(response, dict) else {}
        )
        return error_details.get("Code") in {"404", "NoSuchKey", "NotFound"}

    @staticmethod
    def _is_access_denied(error: Exception) -> bool:
        response = getattr(error, "response", {})
        error_details = (
            response.get("Error", {}) if isinstance(response, dict) else {}
        )
        return error_details.get("Code") in {
            "403",
            "AccessDenied",
            "Forbidden",
        }

    @staticmethod
    def _is_write_conflict(error: Exception) -> bool:
        response = getattr(error, "response", {})
        details = response.get("Error", {}) if isinstance(response, dict) else {}
        return details.get("Code") in {
            "409",
            "412",
            "ConditionalRequestConflict",
            "PreconditionFailed",
        }

    def _object_exists(self, key: str) -> bool:
        response = self._s3_client.list_objects_v2(
            Bucket=self._bucket_name,
            Prefix=key,
            MaxKeys=1,
        )
        contents = response.get("Contents", [])
        return any(
            isinstance(item, dict) and item.get("Key") == key
            for item in contents
        )


class FileSystemKnowledgeStorage:
    """Explicit local artifact storage using the canonical knowledge keys."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser().resolve()

    def put_bytes(
        self,
        key: str,
        content: bytes,
        *,
        content_type: str,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        del content_type, metadata
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(path, content)

    def put_json(self, key: str, payload: Mapping[str, Any]) -> None:
        content = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.put_bytes(key, content, content_type="application/json")

    def get_json(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Unable to read JSON object at '{key}'") from error
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected a JSON object at '{key}'")
        return parsed

    def get_json_versioned(self, key: str) -> VersionedJsonObject:
        payload = self.get_json(key)
        if payload is None:
            return VersionedJsonObject(None, None)
        content = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return VersionedJsonObject(payload, hashlib.sha256(content).hexdigest())

    def put_json_if_version(
        self,
        key: str,
        payload: Mapping[str, Any],
        expected_version: str | None,
    ) -> None:
        current = self.get_json_versioned(key)
        if current.version != expected_version:
            raise ConditionalStorageConflictError(
                "Conditional object write conflict"
            )
        self.put_json(key, payload)

    def _path(self, key: str) -> Path:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("storage key cannot be empty")
        if "\\" in key:
            raise ValueError("storage key must use canonical forward slashes")
        parts = key.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise ValueError("storage key cannot escape the configured root")
        candidate = self._root.joinpath(*parts).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError as error:
            raise ValueError("storage key cannot escape the configured root") from error
        return candidate

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                delete=False,
            ) as temporary:
                temporary.write(content)
                temporary.flush()
                temporary_name = temporary.name
            Path(temporary_name).replace(path)
        finally:
            if temporary_name is not None:
                temporary_path = Path(temporary_name)
                if temporary_path.exists():
                    temporary_path.unlink()
