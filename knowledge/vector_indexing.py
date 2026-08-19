"""Automatic, resumable indexing of pending knowledge descriptors."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from time import perf_counter
from typing import Any

from knowledge.embedding_workflow import (
    EmbeddingFailure,
    EmbeddingWorkflow,
    EmbeddingWorkflowReport,
)
from knowledge.embeddings import EmbeddingStatus
from knowledge.indexing_errors import IndexingBatchLimitError
from knowledge.manifest import (
    IndexingManifestUpdate,
    KnowledgeManifestRepository,
)
from knowledge.media_classification import require_indexable_object
from knowledge.models import (
    EmbeddingRecord,
    KnowledgeChunk,
    KnowledgeManifestEntry,
    chunk_text_checksum,
)
from knowledge.retrieval import RetrievalEntry
from knowledge.storage import KnowledgeKeys, KnowledgeStorage
from knowledge.vector_store import (
    VectorIngestionStatus,
    VectorStore,
    normalize_vector_scope,
)


logger = logging.getLogger(__name__)
_PROTECTED_METADATA = frozenset(
    {
        "checksum",
        "client_id",
        "document_id",
        "document_type",
        "domain",
        "environment",
        "knowledge_domain",
        "knowledge_namespace",
        "namespace",
        "object_classification",
        "indexable",
        "storage_only",
        "source",
    }
)


@dataclass(frozen=True)
class IndexingFailure:
    """Sanitized failure for one pending chunk and processing stage."""

    chunk_id: str
    stage: str
    error_type: str


@dataclass(frozen=True)
class VectorIndexingStatistics:
    """Stable counters describing the current document index state."""

    total_chunk_count: int
    newly_indexed_chunk_count: int
    already_indexed_chunk_count: int
    indexed_chunk_count: int
    pending_chunk_count: int
    failed_chunk_count: int
    embedding_created_count: int
    embedding_skipped_count: int


@dataclass(frozen=True)
class VectorIndexingReport:
    """Embedding and vector persistence outcome for one document."""

    document_id: str
    embedding_report: EmbeddingWorkflowReport
    vector_status: VectorIngestionStatus
    upserted_count: int
    vector_store_provider: str
    vector_collection: str | None
    vector_dimension: int | None
    indexed_at: str | None
    failures: tuple[IndexingFailure, ...]
    statistics: VectorIndexingStatistics
    client_id: str | None = None
    environment: str | None = None
    knowledge_namespace: str | None = None
    knowledge_domain: str | None = None
    retry_count: int = 0

    @property
    def succeeded(self) -> bool:
        return self.vector_status == VectorIngestionStatus.COMPLETE


@dataclass(frozen=True)
class VectorIndexingBatchReport:
    """Aggregate result for a finite queue of pending descriptors."""

    reports: tuple[VectorIndexingReport, ...]
    documents_received: int
    documents_complete: int
    documents_incomplete: int
    indexed_chunk_count: int
    pending_chunk_count: int
    failed_chunk_count: int


class VectorIndexingWorkflow:
    """Consume pending descriptors through injected provider abstractions."""

    DESCRIPTOR_SCHEMA_VERSION = 2

    def __init__(
        self,
        *,
        storage: KnowledgeStorage,
        embedding_workflow: EmbeddingWorkflow,
        vector_store: VectorStore,
        manifest: KnowledgeManifestRepository | None = None,
        clock: Callable[[], datetime] | None = None,
        event_logger: logging.Logger | None = None,
        maximum_descriptor_batch_size: int = 100,
        maximum_chunks_per_invocation: int = 10_000,
    ) -> None:
        if maximum_descriptor_batch_size <= 0:
            raise ValueError("maximum_descriptor_batch_size must be positive")
        if maximum_chunks_per_invocation <= 0:
            raise ValueError("maximum_chunks_per_invocation must be positive")
        self._storage = storage
        self._embedding_workflow = embedding_workflow
        self._vector_store = vector_store
        self._manifest = manifest or KnowledgeManifestRepository(storage)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._logger = event_logger or logger
        self._maximum_descriptor_batch_size = maximum_descriptor_batch_size
        self._maximum_chunks_per_invocation = maximum_chunks_per_invocation

    def index_pending_documents(
        self,
        entries: Sequence[KnowledgeManifestEntry],
        *,
        client_id: str,
        environment: str,
        knowledge_namespace: str = "default",
        knowledge_domain: str = "general",
        metadata: Mapping[str, Any] | None = None,
    ) -> VectorIndexingBatchReport:
        """Process a queue without special-casing an empty queue."""

        if len(entries) > self._maximum_descriptor_batch_size:
            raise IndexingBatchLimitError(
                "Pending descriptor batch exceeds the configured limit"
            )
        normalize_vector_scope(client_id, environment)
        self._metadata_value(
            knowledge_namespace,
            "knowledge_namespace",
            default="default",
        )
        self._metadata_value(
            knowledge_domain,
            "knowledge_domain",
            default="general",
        )
        reports = tuple(
            self.index_pending_document(
                entry,
                client_id=client_id,
                environment=environment,
                knowledge_namespace=knowledge_namespace,
                knowledge_domain=knowledge_domain,
                metadata=metadata,
            )
            for entry in entries
        )
        complete = sum(report.succeeded for report in reports)
        return VectorIndexingBatchReport(
            reports=reports,
            documents_received=len(entries),
            documents_complete=complete,
            documents_incomplete=len(entries) - complete,
            indexed_chunk_count=sum(
                report.statistics.indexed_chunk_count for report in reports
            ),
            pending_chunk_count=sum(
                report.statistics.pending_chunk_count for report in reports
            ),
            failed_chunk_count=sum(
                report.statistics.failed_chunk_count for report in reports
            ),
        )

    def index_pending_document(
        self,
        entry: KnowledgeManifestEntry,
        *,
        client_id: str,
        environment: str,
        knowledge_namespace: str | None = None,
        knowledge_domain: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> VectorIndexingReport:
        """Load one pending descriptor and advance each chunk independently."""

        require_indexable_object(
            entry.metadata.object_classification,
            stage="indexing_chunk_load",
        )
        payload = self._storage.get_json(entry.chunks_key)
        if payload is None:
            raise ValueError("Pending chunk artifact does not exist")
        raw_chunks = payload.get("chunks")
        if not isinstance(raw_chunks, list):
            raise ValueError("Pending chunk artifact is malformed")
        chunks = tuple(self._parse_chunk(value) for value in raw_chunks)
        if len(chunks) > self._maximum_chunks_per_invocation:
            raise IndexingBatchLimitError(
                "Document chunk count exceeds the configured invocation limit"
            )
        if any(chunk.document_id != entry.document_id for chunk in chunks):
            raise ValueError("Pending chunk artifact has the wrong document ID")

        supplied_metadata = dict(metadata or {})
        conflicts = _PROTECTED_METADATA.intersection(supplied_metadata)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(
                "Protected indexing metadata cannot be overridden: "
                f"{names}"
            )
        namespace = self._metadata_value(
            knowledge_namespace,
            "knowledge_namespace",
            default="default",
        )
        domain = self._metadata_value(
            knowledge_domain,
            "knowledge_domain",
            default="general",
        )
        base_metadata: dict[str, Any] = {
            "checksum": entry.metadata.checksum,
            "document_hash": entry.metadata.checksum,
            "document_id": entry.document_id,
            "document_type": entry.metadata.file_type,
            "domain": domain,
            "filename": entry.metadata.filename,
            "ingestion_timestamp": entry.ingestion_timestamp,
            "knowledge_domain": domain,
            "knowledge_namespace": namespace,
            "namespace": namespace,
            "object_classification": entry.metadata.object_classification,
            "indexable": entry.metadata.indexable,
            "storage_only": entry.metadata.storage_only,
            "source": entry.metadata.source,
            "source_object_key": entry.raw_key,
        }
        base_metadata.update(supplied_metadata)
        return self._index_document(
            document_id=entry.document_id,
            chunks=chunks,
            descriptor_key=entry.embedding_key,
            source_object_key=entry.raw_key,
            client_id=client_id,
            environment=environment,
            namespace=namespace,
            domain=domain,
            metadata=base_metadata,
        )

    def index_document(
        self,
        *,
        document_id: str,
        chunks: Sequence[KnowledgeChunk],
        source_object_key: str,
        client_id: str,
        environment: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> VectorIndexingReport:
        """Compatibility entry point for already-loaded chunk artifacts."""

        raw_metadata = dict(metadata or {})
        checksum = raw_metadata.get("checksum") or raw_metadata.get(
            "document_hash"
        )
        document_type = raw_metadata.get("document_type")
        source = raw_metadata.get("source")
        object_classification = raw_metadata.get("object_classification")
        for name, value in (
            ("checksum", checksum),
            ("document_type", document_type),
            ("source", source),
            ("object_classification", object_classification),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"index_document metadata requires non-empty {name}"
                )
        namespace = self._metadata_value(
            raw_metadata.get("knowledge_namespace")
            or raw_metadata.get("namespace"),
            "knowledge_namespace",
            default="default",
        )
        domain = self._metadata_value(
            raw_metadata.get("knowledge_domain") or raw_metadata.get("domain"),
            "knowledge_domain",
            default="general",
        )
        raw_metadata.update(
            {
                "checksum": checksum.strip(),
                "document_hash": checksum.strip(),
                "document_id": document_id,
                "document_type": document_type.strip(),
                "domain": domain,
                "knowledge_domain": domain,
                "knowledge_namespace": namespace,
                "namespace": namespace,
                "object_classification": str(object_classification).strip(),
                "source": source.strip(),
                "source_object_key": source_object_key,
            }
        )
        return self._index_document(
            document_id=document_id,
            chunks=tuple(chunks),
            descriptor_key=KnowledgeKeys.embeddings(document_id),
            source_object_key=source_object_key,
            client_id=client_id,
            environment=environment,
            namespace=namespace,
            domain=domain,
            metadata=raw_metadata,
        )

    def _index_document(
        self,
        *,
        document_id: str,
        chunks: Sequence[KnowledgeChunk],
        descriptor_key: str,
        source_object_key: str,
        client_id: str,
        environment: str,
        namespace: str,
        domain: str,
        metadata: Mapping[str, Any],
    ) -> VectorIndexingReport:
        started = perf_counter()
        require_indexable_object(
            metadata.get("object_classification", ""),
            stage="vector_indexing",
        )
        scope_client, scope_environment = normalize_vector_scope(
            client_id,
            environment,
        )
        descriptor = self._load_descriptor(descriptor_key, document_id)
        collection = getattr(self._vector_store, "collection_name", None)
        model_id = self._embedding_workflow.model_id
        provider_name = self._embedding_workflow.provider_name
        store_name = self._vector_store.provider_name
        states, reset_dimension = self._reconcile_states(
            descriptor,
            chunks,
            client_id=scope_client,
            environment=scope_environment,
            namespace=namespace,
            domain=domain,
            embedding_model=model_id,
            embedding_provider=provider_name,
            vector_store=store_name,
        )
        already_indexed = sum(
            state["status"] == "indexed" for state in states.values()
        )
        pending_chunks = tuple(
            chunk
            for chunk in chunks
            if states[chunk.chunk_id]["status"] != "indexed"
        )
        for chunk in pending_chunks:
            state = states[chunk.chunk_id]
            state["attempt_count"] = int(state["attempt_count"]) + 1
            state["last_error_type"] = None
            state["last_failure_stage"] = None

        expected_dimension = (
            None if reset_dimension else self._descriptor_dimension(descriptor)
        )
        if pending_chunks:
            embedding_report = self._embedding_workflow.embed_document(
                document_id=document_id,
                chunks=pending_chunks,
                source_object_key=source_object_key,
                object_classification=str(
                    metadata.get("object_classification", "")
                ),
                expected_dimensions=expected_dimension,
            )
        else:
            embedding_report = EmbeddingWorkflowReport(
                document_id=document_id,
                model_id=model_id,
                provider_name=provider_name,
                created=(),
                skipped_chunk_ids=(),
                failures=(),
            )

        failures: list[IndexingFailure] = [
            self._embedding_failure(failure)
            for failure in embedding_report.failures
        ]
        failed_ids = {failure.chunk_id for failure in failures}
        newly_indexed = 0
        run_timestamp: str | None = None

        for chunk in pending_chunks:
            if chunk.chunk_id in failed_ids:
                states[chunk.chunk_id]["last_error_type"] = next(
                    failure.error_type
                    for failure in failures
                    if failure.chunk_id == chunk.chunk_id
                )
                states[chunk.chunk_id][
                    "last_failure_stage"
                ] = "embedding_provider"
                continue
            stage = "embedding_validation"
            try:
                retrieval_entry = self._current_entry(
                    chunk,
                    embedding_model_id=model_id,
                    embedding_provider=provider_name,
                    client_id=scope_client,
                    environment=scope_environment,
                    metadata=metadata,
                )
                dimension = retrieval_entry.embedding_record.embedding_dimensions
                if expected_dimension is None:
                    expected_dimension = dimension
                if dimension != expected_dimension:
                    raise ValueError(
                        "Embedding dimensions changed within one document"
                    )
                stage = "vector_store"
                require_indexable_object(
                    metadata.get("object_classification", ""),
                    stage="vector_store_upsert",
                )
                upserted = self._vector_store.upsert(
                    (retrieval_entry,),
                    client_id=scope_client,
                    environment=scope_environment,
                )
                if upserted != 1:
                    raise ValueError("Vector store returned an invalid upsert count")
            except Exception as error:
                states[chunk.chunk_id]["last_error_type"] = type(error).__name__
                states[chunk.chunk_id]["last_failure_stage"] = stage
                failures.append(
                    IndexingFailure(
                        chunk_id=chunk.chunk_id,
                        stage=stage,
                        error_type=type(error).__name__,
                    )
                )
                continue

            run_timestamp = run_timestamp or self._timestamp()
            state = states[chunk.chunk_id]
            state["status"] = "indexed"
            state["indexed_at"] = run_timestamp
            state["last_error_type"] = None
            state["last_failure_stage"] = None
            newly_indexed += 1

        indexed_count = sum(
            state["status"] == "indexed" for state in states.values()
        )
        pending_count = len(states) - indexed_count
        failed_count = sum(
            state["status"] != "indexed" and state["last_error_type"] is not None
            for state in states.values()
        )
        status = self._status(
            indexed_count=indexed_count,
            pending_count=pending_count,
            failed_count=failed_count,
        )
        previous_indexed_at = descriptor.get("indexed_at")
        indexed_at = (
            (
                run_timestamp
                or self._optional_text(previous_indexed_at)
                or self._timestamp()
            )
            if status == VectorIngestionStatus.COMPLETE
            else None
        )
        descriptor_payload = {
            "schema_version": self.DESCRIPTOR_SCHEMA_VERSION,
            "document_id": document_id,
            "provider": provider_name,
            "status": (
                EmbeddingStatus.COMPLETE.value
                if not embedding_report.failures
                else EmbeddingStatus.FAILED.value
            ),
            "index_status": status.value,
            "embedding_model": model_id,
            "embedding_provider": provider_name,
            "vector_store": store_name,
            "vector_dimension": expected_dimension,
            "vector_collection": collection,
            "client_id": scope_client,
            "environment": scope_environment,
            "namespace": namespace,
            "domain": domain,
            "indexed_at": indexed_at,
            "chunks": [states[chunk.chunk_id] for chunk in chunks],
            # Retain the original descriptor member for compatible readers.
            "vectors": [],
        }
        self._storage.put_json(descriptor_key, descriptor_payload)
        embedding_status = (
            EmbeddingStatus.COMPLETE.value
            if not embedding_report.failures
            else EmbeddingStatus.FAILED.value
        )
        self._manifest.update_indexing_status(
            document_id,
            IndexingManifestUpdate(
                embedding_status=embedding_status,
                index_status=status.value,
                indexed_at=indexed_at,
                embedding_model=model_id,
                embedding_provider=provider_name,
                vector_store=store_name,
                vector_dimension=expected_dimension,
                indexed_chunk_count=indexed_count,
                pending_chunk_count=pending_count,
                failed_chunk_count=failed_count,
                vector_collection=collection,
            ),
        )
        statistics = VectorIndexingStatistics(
            total_chunk_count=len(states),
            newly_indexed_chunk_count=newly_indexed,
            already_indexed_chunk_count=already_indexed,
            indexed_chunk_count=indexed_count,
            pending_chunk_count=pending_count,
            failed_chunk_count=failed_count,
            embedding_created_count=len(embedding_report.created),
            embedding_skipped_count=len(embedding_report.skipped_chunk_ids),
        )
        report = VectorIndexingReport(
            document_id=document_id,
            embedding_report=embedding_report,
            vector_status=status,
            upserted_count=newly_indexed,
            vector_store_provider=store_name,
            vector_collection=collection,
            vector_dimension=expected_dimension,
            indexed_at=indexed_at,
            failures=tuple(failures),
            statistics=statistics,
            client_id=scope_client,
            environment=scope_environment,
            knowledge_namespace=namespace,
            knowledge_domain=domain,
            retry_count=max(
                (int(state["attempt_count"]) - 1 for state in states.values()),
                default=0,
            ),
        )
        self._log_report(report, elapsed_seconds=perf_counter() - started)
        return report

    def _current_entry(
        self,
        chunk: KnowledgeChunk,
        *,
        embedding_model_id: str,
        embedding_provider: str,
        client_id: str,
        environment: str,
        metadata: Mapping[str, Any],
    ) -> RetrievalEntry:
        payload = self._storage.get_json(
            KnowledgeKeys.embedding_record(chunk.document_id, chunk.chunk_id)
        )
        if payload is None:
            raise ValueError("Current embedding record does not exist")
        record = EmbeddingRecord.from_dict(payload)
        checksum = chunk_text_checksum(chunk.text)
        if (
            record.document_id != chunk.document_id
            or record.chunk_id != chunk.chunk_id
            or record.chunk_text_checksum != checksum
            or record.embedding_model_id != embedding_model_id
            or record.embedding_provider not in {None, embedding_provider}
        ):
            raise ValueError("Current embedding record does not match its chunk")
        entry_metadata = dict(metadata)
        entry_metadata.update(
            {
                "client_id": client_id,
                "environment": environment,
                "chunk_index": chunk.index,
                "embedding_model": record.embedding_model_id,
                "embedding_provider": embedding_provider,
            }
        )
        return RetrievalEntry(
            embedding_record=record,
            source=str(entry_metadata.get("source") or record.source_object_key),
            text=chunk.text,
            metadata=entry_metadata,
        )

    def _load_descriptor(
        self,
        descriptor_key: str,
        document_id: str,
    ) -> dict[str, Any]:
        descriptor = self._storage.get_json(descriptor_key)
        if descriptor is None:
            raise ValueError("Pending embedding descriptor does not exist")
        if descriptor.get("document_id") != document_id:
            raise ValueError("Pending embedding descriptor has the wrong document ID")
        return descriptor

    def _reconcile_states(
        self,
        descriptor: Mapping[str, Any],
        chunks: Sequence[KnowledgeChunk],
        *,
        client_id: str,
        environment: str,
        namespace: str,
        domain: str,
        embedding_model: str,
        embedding_provider: str,
        vector_store: str,
    ) -> tuple[dict[str, dict[str, Any]], bool]:
        stored_client = self._optional_text(descriptor.get("client_id"))
        stored_environment = self._optional_text(descriptor.get("environment"))
        if stored_client is not None and stored_client != client_id:
            raise ValueError("Pending descriptor belongs to another client")
        if stored_environment is not None and stored_environment != environment:
            raise ValueError("Pending descriptor belongs to another environment")

        signature = {
            "embedding_model": embedding_model,
            "embedding_provider": embedding_provider,
            "vector_store": vector_store,
            "namespace": namespace,
            "domain": domain,
        }
        reset = any(
            self._optional_text(descriptor.get(key)) not in {None, value}
            for key, value in signature.items()
        )
        reset_dimension = any(
            self._optional_text(descriptor.get(key)) not in {None, value}
            for key, value in {
                "embedding_model": embedding_model,
                "embedding_provider": embedding_provider,
            }.items()
        )
        raw_states = descriptor.get("chunks")
        indexed_by_id: dict[str, Mapping[str, Any]] = {}
        if isinstance(raw_states, list):
            for value in raw_states:
                if isinstance(value, Mapping):
                    chunk_id = self._optional_text(value.get("chunk_id"))
                    if chunk_id is not None and chunk_id not in indexed_by_id:
                        indexed_by_id[chunk_id] = value

        states: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            checksum = chunk_text_checksum(chunk.text)
            previous = indexed_by_id.get(chunk.chunk_id, {})
            previous_checksum = self._optional_text(previous.get("checksum"))
            is_indexed = (
                not reset
                and previous.get("status") == "indexed"
                and previous_checksum == checksum
            )
            attempt_count = previous.get("attempt_count", 0)
            if isinstance(attempt_count, bool) or not isinstance(attempt_count, int):
                attempt_count = 0
            states[chunk.chunk_id] = {
                "attempt_count": max(attempt_count, 0),
                "checksum": checksum,
                "chunk_id": chunk.chunk_id,
                "indexed_at": (
                    self._optional_text(previous.get("indexed_at"))
                    if is_indexed
                    else None
                ),
                "last_error_type": None,
                "last_failure_stage": None,
                "status": "indexed" if is_indexed else "pending",
            }
        return states, reset_dimension

    @staticmethod
    def _status(
        *,
        indexed_count: int,
        pending_count: int,
        failed_count: int,
    ) -> VectorIngestionStatus:
        if pending_count == 0:
            return VectorIngestionStatus.COMPLETE
        if indexed_count:
            return VectorIngestionStatus.PARTIAL
        if failed_count:
            return VectorIngestionStatus.FAILED
        return VectorIngestionStatus.PENDING

    @staticmethod
    def _embedding_failure(failure: EmbeddingFailure) -> IndexingFailure:
        return IndexingFailure(
            chunk_id=failure.chunk_id,
            stage="embedding_provider",
            error_type=failure.error_type,
        )

    def _timestamp(self) -> str:
        timestamp = self._clock()
        if timestamp.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _descriptor_dimension(descriptor: Mapping[str, Any]) -> int | None:
        value = descriptor.get("vector_dimension")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None
        return value

    @staticmethod
    def _optional_text(value: object) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _metadata_value(
        value: object,
        name: str,
        *,
        default: str,
    ) -> str:
        selected = default if value is None else value
        if not isinstance(selected, str) or not selected.strip():
            raise ValueError(f"{name} cannot be empty")
        return selected.strip()

    def _log_report(
        self,
        report: VectorIndexingReport,
        *,
        elapsed_seconds: float,
    ) -> None:
        statistics = report.statistics
        event = {
            "client_id": report.client_id,
            "document_id": report.document_id,
            "domain": report.knowledge_domain,
            "elapsed_ms": round(elapsed_seconds * 1_000, 3),
            "embedding_model": report.embedding_report.model_id,
            "embedding_provider": report.embedding_report.provider_name,
            "event": "automatic_vector_indexing",
            "failed_chunk_count": statistics.failed_chunk_count,
            "index_status": report.vector_status.value,
            "indexed_chunk_count": statistics.indexed_chunk_count,
            "environment": report.environment,
            "failure_stage": (
                report.failures[0].stage if report.failures else None
            ),
            "failure_type": (
                report.failures[0].error_type if report.failures else None
            ),
            "namespace": report.knowledge_namespace,
            "pending_chunk_count": statistics.pending_chunk_count,
            "retry_count": report.retry_count,
            "total_chunk_count": statistics.total_chunk_count,
            "vector_store": report.vector_store_provider,
        }
        self._logger.info(json.dumps(event, sort_keys=True))

    @staticmethod
    def _parse_chunk(payload: Any) -> KnowledgeChunk:
        if not isinstance(payload, Mapping):
            raise ValueError("Pending chunk artifact is malformed")
        try:
            chunk = KnowledgeChunk(
                chunk_id=str(payload["chunk_id"]),
                document_id=str(payload["document_id"]),
                index=int(payload["index"]),
                text=str(payload["text"]),
                start_character=int(payload["start_character"]),
                end_character=int(payload["end_character"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Pending chunk artifact is malformed") from error
        if (
            not chunk.chunk_id
            or not chunk.document_id
            or chunk.index < 0
            or chunk.start_character < 0
            or chunk.end_character < chunk.start_character
        ):
            raise ValueError("Pending chunk artifact is malformed")
        return chunk
