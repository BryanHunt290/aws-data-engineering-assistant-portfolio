"""Qdrant adapter for scoped provider-neutral vector storage."""

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any
from urllib.parse import urlparse
import uuid

from knowledge.retrieval import RetrievalEntry, RetrievalResult
from knowledge.media_classification import require_indexable_metadata
from knowledge.vector_store import normalize_vector_scope
from knowledge.vector_store_errors import (
    VectorCollectionConfigurationError,
    VectorDimensionMismatchError,
    VectorRetrievalError,
    VectorStoreUnavailableError,
    VectorUpsertError,
)


_POINT_NAMESPACE = uuid.UUID("40f39c64-8088-4ff0-9739-fd2b37310ddd")
_SENSITIVE_KEY = re.compile(
    r"(?:api.?key|access.?token|password|secret|credential|ssn|"
    r"social.?security|account.?number|card.?number)",
    re.IGNORECASE,
)
_SSN = re.compile(r"(?<!\d)\d{3}-?\d{2}-?\d{4}(?!\d)")
_ACCOUNT_OR_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")


class QdrantVectorStore:
    """Store and query dense vectors with mandatory payload isolation."""

    provider_name = "qdrant"
    supported_filter_fields = frozenset(
        {
            "agent",
            "document_type",
            "knowledge_domain",
            "knowledge_namespace",
            "source",
        }
    )

    def __init__(
        self,
        *,
        url: str = "http://localhost:6333",
        collection_name: str = "dea_knowledge_embeddinggemma_v1",
        api_key: str | None = None,
        timeout_seconds: float = 10.0,
        client: Any | None = None,
        models_module: Any | None = None,
    ) -> None:
        self.url = self._validate_url(url, api_key)
        self.collection_name = self._validate_collection_name(
            collection_name
        )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self._api_key = api_key.strip() if api_key and api_key.strip() else None
        self._client = client
        self._models_module = models_module

    def upsert(
        self,
        entries: Sequence[RetrievalEntry],
        *,
        client_id: str,
        environment: str,
    ) -> int:
        scope_client, scope_environment = normalize_vector_scope(
            client_id,
            environment,
        )
        if not entries:
            return 0
        for entry in entries:
            require_indexable_metadata(
                entry.metadata,
                stage="vector_store_upsert",
            )
        dimensions = {entry.embedding_record.embedding_dimensions for entry in entries}
        if len(dimensions) != 1:
            raise VectorDimensionMismatchError(
                "All vectors in one upsert must have the same dimensions"
            )
        vector_size = dimensions.pop()
        self._ensure_collection(vector_size)
        models = self._models()
        points = []
        for entry in entries:
            self._validate_entry_scope(
                entry,
                client_id=scope_client,
                environment=scope_environment,
            )
            points.append(
                models.PointStruct(
                    id=self.deterministic_point_id(
                        entry,
                        client_id=scope_client,
                        environment=scope_environment,
                    ),
                    vector=list(entry.embedding_record.embedding_vector),
                    payload=self._build_payload(
                        entry,
                        client_id=scope_client,
                        environment=scope_environment,
                    ),
                )
            )
        try:
            self._get_client().upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
        except Exception as error:
            raise VectorUpsertError("Qdrant vector upsert failed") from error
        return len(points)

    def retrieve(
        self,
        query_vector: Sequence[float],
        *,
        client_id: str,
        environment: str,
        filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
        minimum_similarity: float | None = None,
    ) -> list[RetrievalResult]:
        scope_client, scope_environment = normalize_vector_scope(
            client_id,
            environment,
        )
        query = self._validate_vector(query_vector)
        limit = 5 if top_k is None else top_k
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("top_k must be greater than zero")
        threshold = minimum_similarity
        if threshold is not None and (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not -1.0 <= float(threshold) <= 1.0
        ):
            raise ValueError("minimum_similarity must be between -1 and 1")
        self._ensure_collection(len(query), create_if_missing=False)
        query_filter = self._query_filter(
            client_id=scope_client,
            environment=scope_environment,
            filters=filters,
        )
        try:
            response = self._get_client().query_points(
                collection_name=self.collection_name,
                query=list(query),
                query_filter=query_filter,
                limit=limit,
                score_threshold=(
                    float(threshold) if threshold is not None else None
                ),
                with_payload=True,
                with_vectors=False,
            )
        except VectorCollectionConfigurationError:
            raise
        except Exception as error:
            raise VectorRetrievalError("Qdrant vector retrieval failed") from error
        return self._parse_results(response)

    def check_connection(self) -> None:
        """Perform a bounded, explicit connectivity check."""

        try:
            self._get_client().get_collections()
        except Exception as error:
            raise VectorStoreUnavailableError(
                "The configured Qdrant service is unavailable"
            ) from error

    def deterministic_point_id(
        self,
        entry: RetrievalEntry,
        *,
        client_id: str,
        environment: str,
    ) -> str:
        """Return the stable UUID used for idempotent chunk updates."""

        scope_client, scope_environment = normalize_vector_scope(
            client_id,
            environment,
        )
        record = entry.embedding_record
        namespace, domain = self._entry_isolation_scope(entry)
        identity = "\x1f".join(
            (
                scope_client,
                scope_environment,
                namespace,
                domain,
                record.document_id,
                record.chunk_id,
                record.chunk_text_checksum,
                record.embedding_model_id,
            )
        )
        return str(uuid.uuid5(_POINT_NAMESPACE, identity))

    def _ensure_collection(
        self,
        vector_size: int,
        *,
        create_if_missing: bool = True,
    ) -> None:
        client = self._get_client()
        try:
            exists = client.collection_exists(self.collection_name)
        except Exception as error:
            raise VectorStoreUnavailableError(
                "The configured Qdrant service is unavailable"
            ) from error
        if not exists:
            if not create_if_missing:
                raise VectorCollectionConfigurationError(
                    "The configured Qdrant collection does not exist"
                )
            try:
                models = self._models()
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
            except Exception as error:
                raise VectorCollectionConfigurationError(
                    "Qdrant collection creation failed"
                ) from error
            return
        try:
            collection = client.get_collection(self.collection_name)
            configured_size, distance = self._collection_vector_config(
                collection
            )
        except VectorCollectionConfigurationError:
            raise
        except Exception as error:
            raise VectorCollectionConfigurationError(
                "Qdrant collection configuration could not be read"
            ) from error
        if configured_size != vector_size:
            raise VectorDimensionMismatchError(
                "Qdrant collection vector dimensions are incompatible with "
                "the selected embedding model"
            )
        if distance.casefold() != "cosine":
            raise VectorCollectionConfigurationError(
                "Qdrant collection must use cosine distance"
            )

    @staticmethod
    def _collection_vector_config(collection: Any) -> tuple[int, str]:
        vectors = collection.config.params.vectors
        if isinstance(vectors, Mapping):
            raise VectorCollectionConfigurationError(
                "Named Qdrant vectors are not supported by this adapter"
            )
        size = getattr(vectors, "size", None)
        distance = getattr(vectors, "distance", None)
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or distance is None
        ):
            raise VectorCollectionConfigurationError(
                "Qdrant collection vector configuration is malformed"
            )
        distance_value = getattr(distance, "value", distance)
        return size, str(distance_value)

    def _query_filter(
        self,
        *,
        client_id: str,
        environment: str,
        filters: Mapping[str, Any] | None,
    ) -> Any:
        models = self._models()
        conditions = [
            models.FieldCondition(
                key="client_id",
                match=models.MatchValue(value=client_id),
            ),
            models.FieldCondition(
                key="environment",
                match=models.MatchValue(value=environment),
            ),
        ]
        for key, value in dict(filters or {}).items():
            if key not in self.supported_filter_fields:
                raise ValueError(f"Unsupported vector payload filter '{key}'")
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError("Vector payload filter values must be scalar")
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            )
        return models.Filter(must=conditions)

    def _build_payload(
        self,
        entry: RetrievalEntry,
        *,
        client_id: str,
        environment: str,
    ) -> dict[str, Any]:
        record = entry.embedding_record
        original = self._sanitize_metadata(entry.metadata)
        namespace, domain = self._entry_isolation_scope(entry)
        payload: dict[str, Any] = {
            "client_id": client_id,
            "environment": environment,
            "document_id": record.document_id,
            "chunk_id": record.chunk_id,
            "text": self._redact_sensitive_text(entry.text),
            "source": entry.source,
            "source_key": record.source_object_key,
            "checksum": original.get("checksum")
            or original.get("document_hash"),
            "document_hash": original.get("document_hash"),
            "chunk_index": original.get("chunk_index"),
            "page": original.get("page"),
            "document_type": original.get("document_type")
            or original.get("file_type"),
            "embedding_model": record.embedding_model_id,
            "ingestion_timestamp": record.creation_timestamp,
            "knowledge_domain": domain,
            "knowledge_namespace": namespace,
            "domain": domain,
            "namespace": namespace,
            "agent": original.get("agent"),
            "original_metadata": original,
        }
        return {
            key: value
            for key, value in payload.items()
            if value is not None
        }

    @classmethod
    def _sanitize_metadata(cls, metadata: Mapping[str, Any]) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}
        for key, value in metadata.items():
            name = str(key)
            if _SENSITIVE_KEY.search(name):
                continue
            if isinstance(value, str):
                sanitized[name] = cls._redact_sensitive_text(value)
            elif value is None or isinstance(value, (bool, int, float)):
                sanitized[name] = value
            elif isinstance(value, Mapping):
                sanitized[name] = cls._sanitize_metadata(value)
            elif isinstance(value, (list, tuple)):
                sanitized[name] = [
                    cls._redact_sensitive_text(item)
                    if isinstance(item, str)
                    else item
                    for item in value
                    if item is None or isinstance(item, (str, bool, int, float))
                ]
        return sanitized

    @staticmethod
    def _redact_sensitive_text(text: str) -> str:
        redacted = _SSN.sub("[REDACTED-SSN]", text)
        return _ACCOUNT_OR_CARD.sub("[REDACTED-ACCOUNT]", redacted)

    @staticmethod
    def _validate_entry_scope(
        entry: RetrievalEntry,
        *,
        client_id: str,
        environment: str,
    ) -> None:
        entry_client, entry_environment = normalize_vector_scope(
            str(entry.metadata.get("client_id", "")),
            str(entry.metadata.get("environment", "")),
        )
        if entry_client != client_id or entry_environment != environment:
            raise ValueError("Vector entry does not match its client scope")
        QdrantVectorStore._entry_isolation_scope(entry)

    @staticmethod
    def _entry_isolation_scope(entry: RetrievalEntry) -> tuple[str, str]:
        metadata = entry.metadata
        resolved: list[str] = []
        for primary, alias in (
            ("knowledge_namespace", "namespace"),
            ("knowledge_domain", "domain"),
        ):
            primary_value = metadata.get(primary)
            alias_value = metadata.get(alias)
            values = [
                value.strip()
                for value in (primary_value, alias_value)
                if isinstance(value, str) and value.strip()
            ]
            if not values:
                raise ValueError(
                    "Vector entry requires namespace and domain isolation"
                )
            if len(set(values)) != 1:
                raise ValueError(
                    "Vector entry namespace or domain aliases conflict"
                )
            resolved.append(values[0])
        return resolved[0], resolved[1]

    @staticmethod
    def _validate_vector(vector: Sequence[float]) -> tuple[float, ...]:
        if not vector:
            raise ValueError("Vector cannot be empty")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in vector
        ):
            raise ValueError("Vector must contain only numbers")
        parsed = tuple(float(value) for value in vector)
        if not all(math.isfinite(value) for value in parsed):
            raise ValueError("Vector contains a non-finite value")
        return parsed

    @staticmethod
    def _parse_results(response: Any) -> list[RetrievalResult]:
        points = getattr(response, "points", None)
        if not isinstance(points, list):
            raise VectorRetrievalError(
                "Qdrant returned a malformed retrieval response"
            )
        results: list[RetrievalResult] = []
        try:
            for point in points:
                payload = point.payload
                if not isinstance(payload, Mapping):
                    raise ValueError("payload must be an object")
                score = point.score
                if (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(float(score))
                ):
                    raise ValueError("score must be finite")
                document_id = payload["document_id"]
                chunk_id = payload["chunk_id"]
                source = payload["source"]
                chunk_text = payload["text"]
                if not all(
                    isinstance(value, str) and value.strip()
                    for value in (document_id, chunk_id, source, chunk_text)
                ):
                    raise ValueError("required payload fields are invalid")
                original = payload.get("original_metadata", {})
                metadata = (
                    dict(original) if isinstance(original, Mapping) else {}
                )
                metadata.update(
                    {
                        key: value
                        for key, value in payload.items()
                        if key not in {"text", "original_metadata"}
                    }
                )
                results.append(
                    RetrievalResult(
                        document_id=document_id,
                        chunk_id=chunk_id,
                        source=source,
                        text=chunk_text,
                        similarity_score=float(score),
                        metadata=metadata,
                    )
                )
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise VectorRetrievalError(
                "Qdrant returned a malformed retrieval response"
            ) from error
        return results

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as error:
                raise VectorCollectionConfigurationError(
                    "qdrant-client is required for the qdrant provider"
                ) from error
            self._client = QdrantClient(
                url=self.url,
                api_key=self._api_key,
                timeout=self.timeout_seconds,
            )
        return self._client

    def _models(self) -> Any:
        if self._models_module is None:
            try:
                from qdrant_client import models
            except ImportError as error:
                raise VectorCollectionConfigurationError(
                    "qdrant-client is required for the qdrant provider"
                ) from error
            self._models_module = models
        return self._models_module

    @staticmethod
    def _validate_url(value: str, api_key: str | None) -> str:
        if not isinstance(value, str):
            raise ValueError("url must be a string")
        normalized = value.strip().rstrip("/")
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise ValueError("url must be an absolute HTTP or HTTPS URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("url must not contain credentials or query data")
        is_loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if not is_loopback and parsed.scheme != "https":
            raise ValueError("Non-loopback Qdrant URLs must use HTTPS")
        if not is_loopback and not (api_key and api_key.strip()):
            raise ValueError("Non-loopback Qdrant URLs require an API key")
        return normalized

    @staticmethod
    def _validate_collection_name(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("collection_name must be a string")
        normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        if not normalized:
            raise ValueError("collection_name must contain letters or numbers")
        if len(normalized) > 120:
            raise ValueError("collection_name is too long")
        return normalized
