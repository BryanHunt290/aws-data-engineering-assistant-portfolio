"""Typed models used by the knowledge layer."""

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any


def chunk_text_checksum(text: str) -> str:
    """Return the canonical checksum for an immutable chunk body."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DocumentMetadata:
    """Metadata extracted before a document enters processing."""

    filename: str
    file_type: str
    upload_timestamp: str
    checksum: str
    source: str
    document_size: int
    object_classification: str = "indexable_text_document"
    detected_mime_type: str | None = None
    declared_mime_type: str | None = None
    file_extension: str | None = None
    media_type: str | None = None
    storage_only: bool = False
    indexable: bool = True
    quarantine_reason: str | None = None
    source_s3_uri: str | None = None
    checksum_sha256: str | None = None
    size_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocumentMetadata":
        """Parse persisted metadata while rejecting incomplete records."""

        try:
            return cls(
                filename=str(payload["filename"]),
                file_type=str(payload["file_type"]),
                upload_timestamp=str(payload["upload_timestamp"]),
                checksum=str(payload["checksum"]),
                source=str(payload["source"]),
                document_size=int(payload["document_size"]),
                object_classification=str(
                    payload.get(
                        "object_classification",
                        "indexable_text_document",
                    )
                ),
                detected_mime_type=_optional_string(
                    payload.get("detected_mime_type")
                ),
                declared_mime_type=_optional_string(
                    payload.get("declared_mime_type")
                ),
                file_extension=_optional_string(
                    payload.get("file_extension")
                ),
                media_type=_optional_string(payload.get("media_type")),
                storage_only=bool(payload.get("storage_only", False)),
                indexable=bool(payload.get("indexable", True)),
                quarantine_reason=_optional_string(
                    payload.get("quarantine_reason")
                ),
                source_s3_uri=_optional_string(
                    payload.get("source_s3_uri")
                ),
                checksum_sha256=_optional_string(
                    payload.get("checksum_sha256")
                )
                or str(payload["checksum"]),
                size_bytes=int(
                    payload.get("size_bytes", payload["document_size"])
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Malformed document metadata") from error


@dataclass(frozen=True)
class KnowledgeChunk:
    """A text chunk that retains its source document and character range."""

    chunk_id: str
    document_id: str
    index: int
    text: str
    start_character: int
    end_character: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KnowledgeManifestEntry:
    """Manifest state for one ingested document."""

    document_id: str
    metadata: DocumentMetadata
    chunk_count: int
    embedding_status: str
    ingestion_timestamp: str
    raw_key: str
    processed_key: str | None
    chunks_key: str
    embedding_key: str
    vector_status: str = "pending"
    vector_store_provider: str | None = None
    vector_collection: str | None = None
    indexed_at: str | None = None
    embedding_model: str | None = None
    embedding_provider: str | None = None
    vector_store: str | None = None
    vector_dimension: int | None = None
    index_status: str = "pending"
    indexed_chunk_count: int = 0
    pending_chunk_count: int = 0
    failed_chunk_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = self.metadata.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KnowledgeManifestEntry":
        """Load both legacy and extended manifest entries."""

        try:
            metadata = payload["metadata"]
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be an object")
            return cls(
                document_id=str(payload["document_id"]),
                metadata=DocumentMetadata.from_dict(metadata),
                chunk_count=int(payload["chunk_count"]),
                embedding_status=str(payload["embedding_status"]),
                ingestion_timestamp=str(payload["ingestion_timestamp"]),
                raw_key=str(payload["raw_key"]),
                processed_key=(
                    str(payload["processed_key"])
                    if payload.get("processed_key") is not None
                    else None
                ),
                chunks_key=str(payload["chunks_key"]),
                embedding_key=str(payload["embedding_key"]),
                vector_status=str(payload.get("vector_status", "pending")),
                vector_store_provider=_optional_string(
                    payload.get("vector_store_provider")
                ),
                vector_collection=_optional_string(
                    payload.get("vector_collection")
                ),
                indexed_at=_optional_string(payload.get("indexed_at")),
                embedding_model=_optional_string(
                    payload.get("embedding_model")
                ),
                embedding_provider=_optional_string(
                    payload.get("embedding_provider")
                ),
                vector_store=_optional_string(payload.get("vector_store")),
                vector_dimension=_optional_positive_integer(
                    payload.get("vector_dimension")
                ),
                index_status=str(payload.get("index_status", "pending")),
                indexed_chunk_count=int(
                    payload.get("indexed_chunk_count", 0)
                ),
                pending_chunk_count=int(
                    payload.get("pending_chunk_count", 0)
                ),
                failed_chunk_count=int(payload.get("failed_chunk_count", 0)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Malformed knowledge manifest entry") from error


@dataclass(frozen=True)
class EmbeddingRecord:
    """Versioned persisted vector for one immutable chunk representation."""

    schema_version: int
    document_id: str
    chunk_id: str
    chunk_text_checksum: str
    embedding_model_id: str
    embedding_dimensions: int
    embedding_vector: tuple[float, ...]
    creation_timestamp: str
    source_object_key: str
    embedding_provider: str | None = None

    CURRENT_SCHEMA_VERSION = 1

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["embedding_vector"] = list(self.embedding_vector)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EmbeddingRecord":
        try:
            vector = payload["embedding_vector"]
            if not isinstance(vector, list) or not vector:
                raise ValueError("embedding_vector must be a non-empty list")
            dimensions = payload["embedding_dimensions"]
            if (
                not isinstance(dimensions, int)
                or isinstance(dimensions, bool)
                or dimensions != len(vector)
            ):
                raise ValueError(
                    "embedding_dimensions must match embedding_vector"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                for value in vector
            ):
                raise ValueError("embedding_vector must contain numbers")
            numeric_vector = tuple(float(value) for value in vector)
            if not all(math.isfinite(value) for value in numeric_vector):
                raise ValueError(
                    "embedding_vector must contain finite numbers"
                )
            record = cls(
                schema_version=int(payload["schema_version"]),
                document_id=str(payload["document_id"]),
                chunk_id=str(payload["chunk_id"]),
                chunk_text_checksum=str(
                    payload["chunk_text_checksum"]
                ),
                embedding_model_id=str(payload["embedding_model_id"]),
                embedding_dimensions=dimensions,
                embedding_vector=numeric_vector,
                creation_timestamp=str(payload["creation_timestamp"]),
                source_object_key=str(payload["source_object_key"]),
                embedding_provider=(
                    str(payload["embedding_provider"])
                    if payload.get("embedding_provider") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Malformed embedding record") from error

        if record.schema_version != cls.CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported embedding schema version "
                f"{record.schema_version}"
            )
        return record


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_positive_integer(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Expected a positive integer or None")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("Expected a positive integer or None")
    return parsed
