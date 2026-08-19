"""Incremental orchestration for chunk embedding records."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import math
from time import perf_counter

from knowledge.embeddings import EmbeddingProvider, EmbeddingStatus
from knowledge.manifest import KnowledgeManifestRepository
from knowledge.media_classification import require_indexable_object
from knowledge.models import (
    EmbeddingRecord,
    KnowledgeChunk,
    chunk_text_checksum,
)
from knowledge.storage import KnowledgeKeys, KnowledgeStorage


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingFailure:
    """Sanitized failure information for one chunk."""

    chunk_id: str
    error_type: str
    message: str


@dataclass(frozen=True)
class EmbeddingWorkflowReport:
    """Partial-success result for one document embedding run."""

    document_id: str
    model_id: str
    provider_name: str
    created: tuple[EmbeddingRecord, ...]
    skipped_chunk_ids: tuple[str, ...]
    failures: tuple[EmbeddingFailure, ...]

    @property
    def succeeded(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class _PendingChunk:
    chunk: KnowledgeChunk
    checksum: str
    record_key: str


class EmbeddingWorkflow:
    """Embed stored chunks independently from ingestion and extraction."""

    def __init__(
        self,
        *,
        storage: KnowledgeStorage,
        provider: EmbeddingProvider,
        model_id: str,
        batch_size: int,
        manifest: KnowledgeManifestRepository | None = None,
        clock: Callable[[], datetime] | None = None,
        event_logger: logging.Logger | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id cannot be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self._storage = storage
        self._provider = provider
        self._model_id = model_id
        self._batch_size = batch_size
        self._manifest = manifest
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._logger = event_logger or logger

    @property
    def model_id(self) -> str:
        """Return the configured model identity without exposing the provider."""

        return self._model_id

    @property
    def provider_name(self) -> str:
        """Return the stable provider identity used in persisted state."""

        return self._provider.provider_name

    def embed_document(
        self,
        *,
        document_id: str,
        chunks: Sequence[KnowledgeChunk],
        source_object_key: str,
        object_classification: str,
        expected_dimensions: int | None = None,
    ) -> EmbeddingWorkflowReport:
        """Create missing or stale embedding records for a document."""

        if not document_id:
            raise ValueError("document_id cannot be empty")
        if not source_object_key:
            raise ValueError("source_object_key cannot be empty")
        require_indexable_object(
            object_classification,
            stage="embedding_generation",
        )
        if (
            expected_dimensions is not None
            and (
                isinstance(expected_dimensions, bool)
                or not isinstance(expected_dimensions, int)
                or expected_dimensions <= 0
            )
        ):
            raise ValueError("expected_dimensions must be positive")

        pending: list[_PendingChunk] = []
        skipped: list[str] = []
        failures: list[EmbeddingFailure] = []
        created: list[EmbeddingRecord] = []

        for chunk in chunks:
            if chunk.document_id != document_id:
                raise ValueError(
                    "Every chunk must reference the requested document_id"
                )
            checksum = chunk_text_checksum(chunk.text)
            record_key = KnowledgeKeys.embedding_record(
                document_id,
                chunk.chunk_id,
            )
            started = perf_counter()
            try:
                existing_payload = self._storage.get_json(record_key)
                existing = (
                    EmbeddingRecord.from_dict(existing_payload)
                    if existing_payload is not None
                    else None
                )
            except ValueError:
                existing = None
            except Exception as error:
                failures.append(self._failure(chunk.chunk_id, error))
                self._log_event(
                    document_id=document_id,
                    chunk_id=chunk.chunk_id,
                    elapsed_seconds=perf_counter() - started,
                    outcome="failed",
                    error_type=type(error).__name__,
                )
                continue

            if (
                existing is not None
                and existing.embedding_model_id == self._model_id
                and existing.chunk_text_checksum == checksum
                and existing.embedding_provider
                in {None, self._provider.provider_name}
            ):
                if expected_dimensions is None:
                    expected_dimensions = existing.embedding_dimensions
                if existing.embedding_dimensions == expected_dimensions:
                    skipped.append(chunk.chunk_id)
                    self._log_event(
                        document_id=document_id,
                        chunk_id=chunk.chunk_id,
                        elapsed_seconds=perf_counter() - started,
                        outcome="skipped",
                    )
                    continue
            pending.append(
                _PendingChunk(
                    chunk=chunk,
                    checksum=checksum,
                    record_key=record_key,
                )
            )

        for offset in range(0, len(pending), self._batch_size):
            batch = pending[offset : offset + self._batch_size]
            started = perf_counter()
            try:
                vectors = self._provider.embed(
                    [item.chunk.text for item in batch]
                )
                if len(vectors) != len(batch):
                    raise ValueError(
                        "Embedding provider returned the wrong vector count"
                    )
            except Exception as error:
                elapsed = perf_counter() - started
                for item in batch:
                    failures.append(
                        self._failure(item.chunk.chunk_id, error)
                    )
                    self._log_event(
                        document_id=document_id,
                        chunk_id=item.chunk.chunk_id,
                        elapsed_seconds=elapsed,
                        outcome="failed",
                        error_type=type(error).__name__,
                    )
                continue

            validated: list[tuple[_PendingChunk, list[float]]] = []
            for item, vector in zip(batch, vectors, strict=True):
                try:
                    parsed_vector = self._validate_vector(vector)
                    if expected_dimensions is None:
                        expected_dimensions = len(parsed_vector)
                    if len(parsed_vector) != expected_dimensions:
                        raise ValueError(
                            "Embedding dimensions changed within one document"
                        )
                    validated.append((item, parsed_vector))
                except Exception as error:
                    failures.append(
                        self._failure(item.chunk.chunk_id, error)
                    )
                    self._log_event(
                        document_id=document_id,
                        chunk_id=item.chunk.chunk_id,
                        elapsed_seconds=perf_counter() - started,
                        outcome="failed",
                        error_type=type(error).__name__,
                    )

            for item, parsed_vector in validated:
                item_started = perf_counter()
                try:
                    record = EmbeddingRecord(
                        schema_version=(
                            EmbeddingRecord.CURRENT_SCHEMA_VERSION
                        ),
                        document_id=document_id,
                        chunk_id=item.chunk.chunk_id,
                        chunk_text_checksum=item.checksum,
                        embedding_model_id=self._model_id,
                        embedding_dimensions=len(parsed_vector),
                        embedding_vector=tuple(parsed_vector),
                        creation_timestamp=self._timestamp(),
                        source_object_key=source_object_key,
                        embedding_provider=self._provider.provider_name,
                    )
                    self._storage.put_json(
                        item.record_key,
                        record.to_dict(),
                    )
                except Exception as error:
                    failures.append(
                        self._failure(item.chunk.chunk_id, error)
                    )
                    self._log_event(
                        document_id=document_id,
                        chunk_id=item.chunk.chunk_id,
                        elapsed_seconds=perf_counter() - item_started,
                        outcome="failed",
                        error_type=type(error).__name__,
                    )
                    continue

                created.append(record)
                self._log_event(
                    document_id=document_id,
                    chunk_id=item.chunk.chunk_id,
                    elapsed_seconds=perf_counter() - item_started,
                    outcome="created",
                )

        report = EmbeddingWorkflowReport(
            document_id=document_id,
            model_id=self._model_id,
            provider_name=self._provider.provider_name,
            created=tuple(created),
            skipped_chunk_ids=tuple(skipped),
            failures=tuple(failures),
        )
        if self._manifest is not None:
            status = (
                EmbeddingStatus.COMPLETE
                if report.succeeded
                else EmbeddingStatus.FAILED
            )
            self._manifest.update_embedding_status(
                document_id,
                status.value,
            )
        return report

    def _timestamp(self) -> str:
        timestamp = self._clock()
        if timestamp.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return (
            timestamp.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    @staticmethod
    def _validate_vector(vector: Sequence[float]) -> list[float]:
        if not vector:
            raise ValueError("Embedding vector cannot be empty")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            for value in vector
        ):
            raise ValueError("Embedding vector must contain numbers")
        parsed = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in parsed):
            raise ValueError("Embedding vector contains a non-finite value")
        return parsed

    @staticmethod
    def _failure(chunk_id: str, error: Exception) -> EmbeddingFailure:
        return EmbeddingFailure(
            chunk_id=chunk_id,
            error_type=type(error).__name__,
            message=str(error),
        )

    def _log_event(
        self,
        *,
        document_id: str,
        chunk_id: str,
        elapsed_seconds: float,
        outcome: str,
        error_type: str | None = None,
    ) -> None:
        event: dict[str, object] = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "elapsed_ms": round(elapsed_seconds * 1_000, 3),
            "event": "embedding_workflow",
            "model_id": self._model_id,
            "outcome": outcome,
        }
        if error_type:
            event["error_type"] = error_type
        self._logger.info(json.dumps(event, sort_keys=True))
