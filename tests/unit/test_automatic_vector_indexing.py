from copy import deepcopy
from datetime import datetime, timezone
import logging
from typing import Any, Mapping, Sequence

import pytest

from knowledge.embedding_errors import EmbeddingInvocationError
from knowledge.embedding_workflow import EmbeddingWorkflow
from knowledge.fake_embeddings import DeterministicFakeEmbeddingProvider
from knowledge.indexing_configuration import (
    AutomaticIndexingConfig,
    IndexingEmbeddingProviderName,
    IndexingVectorStoreName,
    build_automatic_indexing_workflow,
)
from knowledge.ingestion import KnowledgeIngestionPipeline
from knowledge.manifest import KnowledgeManifestRepository
from knowledge.retrieval import RetrievalEntry, RetrievalResult
from knowledge.storage import KnowledgeKeys
from knowledge.vector_indexing import VectorIndexingWorkflow
from knowledge.vector_store import (
    InMemoryVectorStore,
    VectorIngestionStatus,
)
from knowledge.vector_store_errors import MissingClientFilterError
from knowledge.config import KnowledgeConfig


FIXED_TIME = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)


class MemoryKnowledgeStorage:
    def __init__(self) -> None:
        self.byte_objects: dict[str, dict[str, Any]] = {}
        self.json_objects: dict[str, dict[str, Any]] = {}

    def put_bytes(
        self,
        key: str,
        content: bytes,
        *,
        content_type: str,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        self.byte_objects[key] = {
            "content": content,
            "content_type": content_type,
            "metadata": dict(metadata or {}),
        }

    def put_json(self, key: str, payload: Mapping[str, Any]) -> None:
        self.json_objects[key] = deepcopy(dict(payload))

    def get_json(self, key: str) -> dict[str, Any] | None:
        payload = self.json_objects.get(key)
        return deepcopy(payload) if payload is not None else None


class StatefulEmbeddingProvider:
    model_id = "shared-model-v1"

    def __init__(
        self,
        *,
        provider_name: str = "stateful-fake",
        failures: set[str] | None = None,
        inconsistent_dimensions: bool = False,
    ) -> None:
        self.provider_name = provider_name
        self.failures = failures or set()
        self.inconsistent_dimensions = inconsistent_dimensions
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(tuple(texts))
        if any(text in self.failures for text in texts):
            raise EmbeddingInvocationError("Synthetic provider failure")
        vectors: list[list[float]] = []
        for text in texts:
            vector = [1.0, float(len(text))]
            if self.inconsistent_dimensions and text.startswith("b"):
                vector.append(0.5)
            vectors.append(vector)
        return vectors


class StatefulVectorStore:
    provider_name = "stateful-store"

    def __init__(self, failures: set[str] | None = None) -> None:
        self.failures = failures or set()
        self.delegate = InMemoryVectorStore(minimum_similarity=-1.0)
        self.upserted_chunk_ids: list[str] = []

    def upsert(
        self,
        entries: Sequence[RetrievalEntry],
        *,
        client_id: str,
        environment: str,
    ) -> int:
        entry = entries[0]
        chunk_id = entry.embedding_record.chunk_id
        if chunk_id in self.failures:
            raise RuntimeError("Synthetic vector-store failure")
        self.upserted_chunk_ids.append(chunk_id)
        return self.delegate.upsert(
            entries,
            client_id=client_id,
            environment=environment,
        )

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
        return self.delegate.retrieve(
            query_vector,
            client_id=client_id,
            environment=environment,
            filters=filters,
            top_k=top_k,
            minimum_similarity=minimum_similarity,
        )


def _ingested(storage: MemoryKnowledgeStorage, content: bytes = b"aaaaabbbbb"):
    return KnowledgeIngestionPipeline(
        storage,
        KnowledgeConfig(chunk_size=5, overlap=0),
        clock=lambda: FIXED_TIME,
        document_id_factory=lambda: "automatic-document",
    ).ingest(
        filename="policy.txt",
        content=content,
        source="synthetic-upload",
    )


def _workflow(
    storage: MemoryKnowledgeStorage,
    provider: Any,
    vector_store: Any,
    *,
    batch_size: int = 1,
    event_logger: logging.Logger | None = None,
) -> VectorIndexingWorkflow:
    manifest = KnowledgeManifestRepository(storage)
    return VectorIndexingWorkflow(
        storage=storage,
        embedding_workflow=EmbeddingWorkflow(
            storage=storage,
            provider=provider,
            model_id=provider.model_id,
            batch_size=batch_size,
            manifest=manifest,
            clock=lambda: FIXED_TIME,
        ),
        vector_store=vector_store,
        manifest=manifest,
        clock=lambda: FIXED_TIME,
        event_logger=event_logger,
    )


def _index(workflow: VectorIndexingWorkflow, entry):
    return workflow.index_pending_document(
        entry,
        client_id="Client A",
        environment="dev",
        knowledge_namespace="policies",
        knowledge_domain="bookkeeping",
        metadata={"authority_level": "synthetic"},
    )


def test_successful_automatic_indexing_preserves_scope_and_updates_manifest():
    storage = MemoryKnowledgeStorage()
    entry = _ingested(storage)
    provider = StatefulEmbeddingProvider()
    store = StatefulVectorStore()

    report = _index(_workflow(storage, provider, store), entry)

    assert report.succeeded
    assert report.vector_dimension == 2
    assert report.statistics.total_chunk_count == 2
    assert report.statistics.newly_indexed_chunk_count == 2
    assert report.statistics.pending_chunk_count == 0
    assert report.failures == ()
    manifest = storage.json_objects[KnowledgeKeys.MANIFEST]["documents"][
        entry.document_id
    ]
    expected_manifest = {
        "indexed_at": "2026-08-01T12:00:00Z",
        "embedding_model": "shared-model-v1",
        "embedding_provider": "stateful-fake",
        "vector_store": "stateful-store",
        "vector_dimension": 2,
        "index_status": "complete",
        "indexed_chunk_count": 2,
        "pending_chunk_count": 0,
        "failed_chunk_count": 0,
    }
    assert {
        key: manifest[key] for key in expected_manifest
    } == expected_manifest
    results = store.retrieve(
        [1.0, 5.0],
        client_id="client-a",
        environment="dev",
        top_k=2,
    )
    assert len(results) == 2
    for result in results:
        expected_metadata = {
            "client_id": "client-a",
            "environment": "dev",
            "namespace": "policies",
            "domain": "bookkeeping",
            "document_id": entry.document_id,
            "document_type": "txt",
            "source": "synthetic-upload",
            "checksum": entry.metadata.checksum,
        }
        assert {
            key: result.metadata[key] for key in expected_metadata
        } == expected_metadata


def test_duplicate_indexing_skips_embedding_and_vector_upsert():
    storage = MemoryKnowledgeStorage()
    entry = _ingested(storage)
    provider = StatefulEmbeddingProvider()
    store = StatefulVectorStore()
    workflow = _workflow(storage, provider, store)

    _index(workflow, entry)
    first_calls = tuple(provider.calls)
    first_upserts = tuple(store.upserted_chunk_ids)
    duplicate = _index(workflow, entry)

    assert tuple(provider.calls) == first_calls
    assert tuple(store.upserted_chunk_ids) == first_upserts
    assert duplicate.upserted_count == 0
    assert duplicate.statistics.already_indexed_chunk_count == 2
    assert duplicate.statistics.embedding_created_count == 0


def test_repeated_ingestion_preserves_descriptor_and_single_manifest_entry():
    storage = MemoryKnowledgeStorage()
    entry = _ingested(storage)
    provider = StatefulEmbeddingProvider()
    store = StatefulVectorStore()
    workflow = _workflow(storage, provider, store)
    _index(workflow, entry)
    call_count = len(provider.calls)
    upsert_count = len(store.upserted_chunk_ids)

    repeated_entry = _ingested(storage)
    repeated = _index(workflow, repeated_entry)

    assert len(
        storage.json_objects[KnowledgeKeys.MANIFEST]["documents"]
    ) == 1
    assert repeated.statistics.already_indexed_chunk_count == 2
    assert len(provider.calls) == call_count
    assert len(store.upserted_chunk_ids) == upsert_count


def test_partial_provider_failure_remains_pending_and_retry_is_incremental():
    storage = MemoryKnowledgeStorage()
    entry = _ingested(storage)
    provider = StatefulEmbeddingProvider(failures={"bbbbb"})
    store = StatefulVectorStore()
    workflow = _workflow(storage, provider, store)

    partial = _index(workflow, entry)
    provider.failures.clear()
    recovered = _index(workflow, entry)

    assert partial.vector_status == VectorIngestionStatus.PARTIAL
    assert partial.statistics.indexed_chunk_count == 1
    assert partial.statistics.pending_chunk_count == 1
    assert partial.statistics.failed_chunk_count == 1
    assert recovered.succeeded
    assert recovered.statistics.newly_indexed_chunk_count == 1
    assert recovered.statistics.already_indexed_chunk_count == 1
    assert store.upserted_chunk_ids.count("automatic-document:000000") == 1
    descriptor = storage.json_objects[entry.embedding_key]
    assert all(state["status"] == "indexed" for state in descriptor["chunks"])


def test_provider_failure_returns_sanitized_statistics_without_upserts():
    storage = MemoryKnowledgeStorage()
    entry = _ingested(storage, b"aaaaa")
    provider = StatefulEmbeddingProvider(failures={"aaaaa"})
    store = StatefulVectorStore()

    report = _index(_workflow(storage, provider, store), entry)

    assert report.vector_status == VectorIngestionStatus.FAILED
    assert report.statistics.failed_chunk_count == 1
    assert report.failures[0].stage == "embedding_provider"
    assert report.failures[0].error_type == "EmbeddingInvocationError"
    assert store.upserted_chunk_ids == []


def test_dimension_mismatch_is_partial_and_never_writes_the_bad_vector():
    storage = MemoryKnowledgeStorage()
    entry = _ingested(storage)
    provider = StatefulEmbeddingProvider(inconsistent_dimensions=True)
    store = StatefulVectorStore()

    report = _index(_workflow(storage, provider, store), entry)

    assert report.vector_status == VectorIngestionStatus.PARTIAL
    assert report.vector_dimension == 2
    assert report.statistics.indexed_chunk_count == 1
    assert report.statistics.failed_chunk_count == 1
    assert store.upserted_chunk_ids == ["automatic-document:000000"]
    provider.inconsistent_dimensions = False
    recovered = _index(_workflow(storage, provider, store), entry)
    assert recovered.succeeded
    assert recovered.statistics.newly_indexed_chunk_count == 1


def test_non_numeric_embedding_is_rejected_before_vector_storage():
    class InvalidNumericProvider(StatefulEmbeddingProvider):
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            self.calls.append(tuple(texts))
            return [[True, 1.0]]  # type: ignore[list-item]

    storage = MemoryKnowledgeStorage()
    entry = _ingested(storage, b"aaaaa")
    store = StatefulVectorStore()

    report = _index(
        _workflow(storage, InvalidNumericProvider(), store),
        entry,
    )

    assert report.vector_status == VectorIngestionStatus.FAILED
    assert report.failures[0].stage == "embedding_provider"
    assert report.failures[0].error_type == "ValueError"
    assert store.upserted_chunk_ids == []


def test_vector_store_failure_preserves_success_and_retries_only_pending_chunk():
    storage = MemoryKnowledgeStorage()
    entry = _ingested(storage)
    provider = StatefulEmbeddingProvider()
    store = StatefulVectorStore(failures={"automatic-document:000001"})
    workflow = _workflow(storage, provider, store)

    partial = _index(workflow, entry)
    store.failures.clear()
    recovered = _index(workflow, entry)

    assert partial.vector_status == VectorIngestionStatus.PARTIAL
    assert partial.failures[0].stage == "vector_store"
    assert recovered.succeeded
    assert recovered.statistics.newly_indexed_chunk_count == 1
    assert store.upserted_chunk_ids.count("automatic-document:000000") == 1


def test_all_vector_store_failures_return_failed_report():
    storage = MemoryKnowledgeStorage()
    entry = _ingested(storage, b"aaaaa")
    store = StatefulVectorStore(failures={"automatic-document:000000"})

    report = _index(
        _workflow(storage, StatefulEmbeddingProvider(), store),
        entry,
    )

    assert report.vector_status == VectorIngestionStatus.FAILED
    assert report.statistics.indexed_chunk_count == 0
    assert report.statistics.pending_chunk_count == 1
    assert report.statistics.failed_chunk_count == 1


def test_missing_client_is_rejected_before_embedding_or_storage_access():
    storage = MemoryKnowledgeStorage()
    entry = _ingested(storage)
    provider = StatefulEmbeddingProvider()
    store = StatefulVectorStore()
    workflow = _workflow(storage, provider, store)

    with pytest.raises(MissingClientFilterError):
        workflow.index_pending_document(
            entry,
            client_id="",
            environment="dev",
        )

    assert provider.calls == []
    assert store.upserted_chunk_ids == []


def test_cross_client_indexing_is_rejected_and_retrieval_stays_isolated():
    storage = MemoryKnowledgeStorage()
    entry = _ingested(storage, b"aaaaa")
    provider = StatefulEmbeddingProvider()
    store = StatefulVectorStore()
    workflow = _workflow(storage, provider, store)
    _index(workflow, entry)
    call_count = len(provider.calls)

    with pytest.raises(ValueError, match="another client"):
        workflow.index_pending_document(
            entry,
            client_id="client-b",
            environment="dev",
            knowledge_namespace="policies",
            knowledge_domain="bookkeeping",
        )

    assert len(provider.calls) == call_count
    assert store.retrieve(
        [1.0, 5.0],
        client_id="client-b",
        environment="dev",
    ) == []


def test_provider_change_with_same_model_reembeds_and_updates_manifest():
    storage = MemoryKnowledgeStorage()
    entry = _ingested(storage, b"aaaaa")
    store = StatefulVectorStore()
    first_provider = StatefulEmbeddingProvider(provider_name="provider-a")
    _index(_workflow(storage, first_provider, store), entry)
    second_provider = StatefulEmbeddingProvider(provider_name="provider-b")

    report = _index(_workflow(storage, second_provider, store), entry)

    assert report.succeeded
    assert second_provider.calls == [("aaaaa",)]
    manifest = storage.json_objects[KnowledgeKeys.MANIFEST]["documents"][
        entry.document_id
    ]
    assert manifest["embedding_provider"] == "provider-b"


def test_empty_pending_queue_returns_zero_statistics_and_validates_scope():
    workflow = _workflow(
        MemoryKnowledgeStorage(),
        StatefulEmbeddingProvider(),
        StatefulVectorStore(),
    )

    report = workflow.index_pending_documents(
        (),
        client_id="client-a",
        environment="dev",
    )

    assert report.documents_received == 0
    assert report.documents_complete == 0
    assert report.documents_incomplete == 0
    assert report.indexed_chunk_count == 0
    with pytest.raises(MissingClientFilterError):
        workflow.index_pending_documents(
            (),
            client_id="",
            environment="dev",
        )


def test_legacy_pending_descriptor_is_consumed_without_migration_step():
    storage = MemoryKnowledgeStorage()
    entry = _ingested(storage, b"aaaaa")
    storage.json_objects[entry.embedding_key] = {
        "document_id": entry.document_id,
        "provider": None,
        "status": "pending",
        "vectors": [],
    }

    report = _index(
        _workflow(
            storage,
            StatefulEmbeddingProvider(),
            StatefulVectorStore(),
        ),
        entry,
    )

    assert report.succeeded
    assert storage.json_objects[entry.embedding_key]["schema_version"] == 2


def test_configuration_is_disabled_by_default_and_supports_offline_fakes():
    storage = MemoryKnowledgeStorage()
    manifest = KnowledgeManifestRepository(storage)
    disabled = AutomaticIndexingConfig.from_environment({})
    assert build_automatic_indexing_workflow(
        disabled,
        storage=storage,
        manifest=manifest,
    ) is None

    configured = AutomaticIndexingConfig(
        enabled=True,
        embedding_provider=IndexingEmbeddingProviderName.FAKE,
        vector_store=IndexingVectorStoreName.MEMORY,
        embedding_model_id="offline-index-v1",
        embedding_dimensions=4,
    )
    assert isinstance(
        build_automatic_indexing_workflow(
            configured,
            storage=storage,
            manifest=manifest,
        ),
        VectorIndexingWorkflow,
    )
    with pytest.raises(ValueError, match="explicit"):
        AutomaticIndexingConfig(enabled=True)


def test_structured_indexing_log_contains_only_identifiers_and_counts(caplog):
    storage = MemoryKnowledgeStorage()
    entry = _ingested(storage, b"private financial text")
    event_logger = logging.getLogger("automatic-indexing-test")
    caplog.set_level(logging.INFO, logger=event_logger.name)

    _index(
        _workflow(
            storage,
            DeterministicFakeEmbeddingProvider(dimensions=4),
            StatefulVectorStore(),
            event_logger=event_logger,
        ),
        entry,
    )

    messages = [record.message for record in caplog.records]
    assert any('"event": "automatic_vector_indexing"' in value for value in messages)
    assert all("private financial text" not in value for value in messages)
