"""Scoped vector-store contract and deterministic in-memory adapter."""

from enum import StrEnum
import re
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from knowledge.retrieval import (
    InMemoryCosineRetriever,
    RetrievalEntry,
    RetrievalResult,
)
from knowledge.media_classification import require_indexable_metadata
from knowledge.vector_store_errors import MissingClientFilterError


_VALID_VECTOR_ENVIRONMENTS = frozenset({"dev", "test", "stage", "prod"})


class VectorIngestionStatus(StrEnum):
    """Manifest states for the optional vector indexing stage."""

    PENDING = "pending"
    PARTIAL = "partial"
    COMPLETE = "complete"
    FAILED = "failed"


@runtime_checkable
class VectorStore(Protocol):
    """Provider-neutral storage and strictly scoped similarity search."""

    @property
    def provider_name(self) -> str:
        """Return a stable vector-store provider identifier."""

    def upsert(
        self,
        entries: Sequence[RetrievalEntry],
        *,
        client_id: str,
        environment: str,
    ) -> int:
        """Insert or update entries within one normalized client scope."""

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
        """Return ranked results after mandatory database-side scoping."""


def normalize_vector_scope(
    client_id: str,
    environment: str,
) -> tuple[str, str]:
    """Normalize vector scope without depending on an interface package."""

    if not isinstance(client_id, str) or not client_id.strip():
        raise MissingClientFilterError(
            "A client identifier is required for vector retrieval"
        )
    try:
        normalized_client = _normalize_scope_value(client_id)
        normalized_environment = _normalize_scope_value(environment)
        if normalized_environment not in _VALID_VECTOR_ENVIRONMENTS:
            raise ValueError("Unsupported vector environment")
        return normalized_client, normalized_environment
    except ValueError as error:
        raise MissingClientFilterError(
            "A valid client and environment are required for vector retrieval"
        ) from error


def _normalize_scope_value(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Vector scope values must be strings")
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    if not normalized:
        raise ValueError("Vector scope values cannot be empty")
    return normalized


class InMemoryVectorStore:
    """Strictly scoped wrapper around the established cosine retriever."""

    provider_name = "memory"

    def __init__(
        self,
        entries: Sequence[RetrievalEntry] = (),
        *,
        top_k: int = 5,
        minimum_similarity: float = 0.0,
    ) -> None:
        self._top_k = top_k
        self._minimum_similarity = minimum_similarity
        self._entries: dict[
            tuple[str, str, str, str], RetrievalEntry
        ] = {}
        if entries:
            scopes = {
                (
                    str(entry.metadata.get("client_id", "")),
                    str(entry.metadata.get("environment", "")),
                )
                for entry in entries
            }
            for client_id, environment in scopes:
                scoped = [
                    entry
                    for entry in entries
                    if entry.metadata.get("client_id") == client_id
                    and entry.metadata.get("environment") == environment
                ]
                self.upsert(
                    scoped,
                    client_id=client_id,
                    environment=environment,
                )

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
        for entry in entries:
            require_indexable_metadata(
                entry.metadata,
                stage="vector_store_upsert",
            )
            metadata = dict(entry.metadata)
            entry_client, entry_environment = normalize_vector_scope(
                str(metadata.get("client_id", "")),
                str(metadata.get("environment", "")),
            )
            if (
                entry_client != scope_client
                or entry_environment != scope_environment
            ):
                raise ValueError("Vector entry does not match its client scope")
            metadata["client_id"] = scope_client
            metadata["environment"] = scope_environment
            normalized_entry = RetrievalEntry(
                embedding_record=entry.embedding_record,
                source=entry.source,
                text=entry.text,
                metadata=metadata,
            )
            # Reuse the established vector validation rather than maintaining
            # a second cosine implementation.
            InMemoryCosineRetriever((normalized_entry,))
            key = (
                scope_client,
                scope_environment,
                entry.embedding_record.document_id,
                entry.embedding_record.chunk_id,
            )
            self._entries[key] = normalized_entry
        return len(entries)

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
        exact_filters = dict(filters or {})
        if any(key in {"client_id", "environment"} for key in exact_filters):
            raise ValueError("Scope fields cannot be overridden by filters")
        scoped_entries = [
            entry
            for (stored_client, stored_environment, _, _), entry
            in self._entries.items()
            if stored_client == scope_client
            and stored_environment == scope_environment
            and all(
                entry.metadata.get(key) == value
                for key, value in exact_filters.items()
            )
        ]
        retriever = InMemoryCosineRetriever(
            scoped_entries,
            top_k=self._top_k,
            minimum_similarity=self._minimum_similarity,
        )
        return retriever.retrieve(
            query_vector,
            top_k=top_k,
            minimum_similarity=minimum_similarity,
        )
