from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Mapping
from unittest.mock import Mock

import pytest
from qdrant_client import QdrantClient, models

from knowledge.application import RAGApplicationService
from knowledge.application_models import ApplicationRequest, ApplicationStatus
from knowledge.classification import RuleBasedIntentClassifier
from knowledge.config import ApplicationConfig, ClassificationRoutingConfig
from knowledge.embedding_workflow import EmbeddingWorkflow
from knowledge.evaluation import RetrievalEvaluationCase, RetrievalEvaluator
from knowledge.fake_embeddings import DeterministicFakeEmbeddingProvider
from knowledge.ingestion import KnowledgeIngestionPipeline
from knowledge.manifest import KnowledgeManifestRepository
from knowledge.models import EmbeddingRecord
from knowledge.qdrant_vector_store import QdrantVectorStore
from knowledge.fake_llm import DeterministicFakeLLMProvider
from knowledge.prompting import GroundedPromptBuilder
from knowledge.retrieval import RetrievalEntry, RetrievalResult
from knowledge.routing import RequestRouter
from knowledge.storage import KnowledgeKeys
from knowledge.vector_indexing import VectorIndexingWorkflow
from knowledge.vector_store import InMemoryVectorStore, VectorIngestionStatus
from knowledge.vector_store_errors import (
    MissingClientFilterError,
    VectorDimensionMismatchError,
    VectorRetrievalError,
    VectorStoreUnavailableError,
    VectorUpsertError,
)


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


def _entry(
    *,
    client_id: str = "client-a",
    environment: str = "dev",
    document_id: str = "document-1",
    chunk_id: str = "document-1:000000",
    vector: tuple[float, ...] = (1.0, 0.0),
    namespace: str = "runbooks",
    domain: str = "data-engineering",
) -> RetrievalEntry:
    record = EmbeddingRecord(
        schema_version=EmbeddingRecord.CURRENT_SCHEMA_VERSION,
        document_id=document_id,
        chunk_id=chunk_id,
        chunk_text_checksum="a" * 64,
        embedding_model_id="fake-v1",
        embedding_dimensions=len(vector),
        embedding_vector=vector,
        creation_timestamp="2026-08-01T12:00:00Z",
        source_object_key=f"knowledge/raw/{document_id}/guide.txt",
    )
    return RetrievalEntry(
        embedding_record=record,
        source="Synthetic guide",
        text="Safe runbook evidence",
        metadata={
            "client_id": client_id,
            "environment": environment,
            "knowledge_namespace": namespace,
            "knowledge_domain": domain,
            "document_hash": "b" * 64,
            "chunk_index": 0,
            "file_type": "txt",
            "object_classification": "indexable_text_document",
            "indexable": True,
            "storage_only": False,
            "api_key": "must-not-store",
            "account_number": "1234567890123456",
        },
    )


def _local_store(client: QdrantClient | None = None) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client or QdrantClient(":memory:"),
        models_module=models,
        collection_name="test_fake_v1",
    )


def test_qdrant_creates_cosine_collection_and_reuses_it():
    client = QdrantClient(":memory:")
    store = _local_store(client)

    assert store.upsert([_entry()], client_id="client-a", environment="dev") == 1
    assert store.upsert([_entry()], client_id="client-a", environment="dev") == 1

    collection = client.get_collection("test_fake_v1")
    assert collection.config.params.vectors.size == 2
    assert collection.config.params.vectors.distance == models.Distance.COSINE
    assert collection.points_count == 1


def test_qdrant_rejects_existing_collection_dimension_mismatch():
    client = QdrantClient(":memory:")
    client.create_collection(
        "test_fake_v1",
        vectors_config=models.VectorParams(
            size=3,
            distance=models.Distance.COSINE,
        ),
    )

    with pytest.raises(VectorDimensionMismatchError, match="dimensions"):
        _local_store(client).upsert(
            [_entry()], client_id="client-a", environment="dev"
        )

    assert client.get_collection("test_fake_v1").config.params.vectors.size == 3


def test_qdrant_point_ids_are_deterministic_and_scope_sensitive():
    store = _local_store()
    entry = _entry()

    first = store.deterministic_point_id(
        entry, client_id="client-a", environment="dev"
    )
    repeated = store.deterministic_point_id(
        entry, client_id="client-a", environment="dev"
    )
    other_client = store.deterministic_point_id(
        entry, client_id="client-b", environment="dev"
    )
    other_namespace = store.deterministic_point_id(
        _entry(namespace="policies"),
        client_id="client-a",
        environment="dev",
    )
    other_domain = store.deterministic_point_id(
        _entry(domain="bookkeeping"),
        client_id="client-a",
        environment="dev",
    )

    assert first == repeated
    assert first != other_client
    assert first != other_namespace
    assert first != other_domain


def test_qdrant_rejects_missing_or_conflicting_isolation_metadata():
    store = _local_store()
    missing_domain = _entry()
    missing_domain.metadata.pop("knowledge_domain")
    conflicting_namespace = _entry()
    conflicting_namespace.metadata["namespace"] = "policies"

    with pytest.raises(ValueError, match="namespace and domain isolation"):
        store.upsert(
            [missing_domain], client_id="client-a", environment="dev"
        )
    with pytest.raises(ValueError, match="aliases conflict"):
        store.upsert(
            [conflicting_namespace],
            client_id="client-a",
            environment="dev",
        )


def test_qdrant_payload_retains_metadata_and_drops_sensitive_keys():
    client = QdrantClient(":memory:")
    store = _local_store(client)
    entry = _entry()
    point_id = store.deterministic_point_id(
        entry, client_id="client-a", environment="dev"
    )

    store.upsert([entry], client_id="client-a", environment="dev")
    point = client.retrieve(
        "test_fake_v1", ids=[point_id], with_payload=True
    )[0]

    assert point.payload["client_id"] == "client-a"
    assert point.payload["environment"] == "dev"
    assert point.payload["embedding_model"] == "fake-v1"
    assert point.payload["knowledge_namespace"] == "runbooks"
    assert point.payload["namespace"] == "runbooks"
    assert point.payload["knowledge_domain"] == "data-engineering"
    assert point.payload["domain"] == "data-engineering"
    assert point.payload["checksum"] == "b" * 64
    original = point.payload["original_metadata"]
    assert "api_key" not in original
    assert "account_number" not in original


def test_qdrant_search_enforces_client_and_namespace_isolation():
    store = _local_store()
    store.upsert(
        [_entry(client_id="client-a")],
        client_id="client-a",
        environment="dev",
    )
    store.upsert(
        [
            _entry(
                client_id="client-b",
                document_id="document-b",
                chunk_id="document-b:000000",
            )
        ],
        client_id="client-b",
        environment="dev",
    )

    client_a = store.retrieve(
        [1.0, 0.0],
        client_id="client-a",
        environment="dev",
        filters={"knowledge_namespace": "runbooks"},
        top_k=5,
        minimum_similarity=0.9,
    )
    wrong_namespace = store.retrieve(
        [1.0, 0.0],
        client_id="client-a",
        environment="dev",
        filters={"knowledge_namespace": "policies"},
        top_k=5,
    )

    assert [result.document_id for result in client_a] == ["document-1"]
    assert wrong_namespace == []


def test_qdrant_search_requires_client_identity():
    with pytest.raises(MissingClientFilterError, match="client"):
        _local_store().retrieve(
            [1.0, 0.0], client_id="", environment="dev"
        )


def test_qdrant_query_passes_mandatory_filters_threshold_and_limit():
    client = Mock()
    client.collection_exists.return_value = True
    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=SimpleNamespace(
                    size=2,
                    distance=models.Distance.COSINE,
                )
            )
        )
    )
    client.query_points.return_value = SimpleNamespace(points=[])
    store = _local_store(client)

    store.retrieve(
        [1.0, 0.0],
        client_id="Client A",
        environment="dev",
        filters={"knowledge_namespace": "runbooks"},
        top_k=3,
        minimum_similarity=0.25,
    )

    request = client.query_points.call_args.kwargs
    assert request["limit"] == 3
    assert request["score_threshold"] == 0.25
    conditions = request["query_filter"].must
    assert [(condition.key, condition.match.value) for condition in conditions] == [
        ("client_id", "client-a"),
        ("environment", "dev"),
        ("knowledge_namespace", "runbooks"),
    ]


def test_qdrant_failures_are_converted_without_payload_leakage():
    unavailable = Mock()
    unavailable.collection_exists.side_effect = RuntimeError("private payload")
    with pytest.raises(VectorStoreUnavailableError) as raised:
        _local_store(unavailable).upsert(
            [_entry()], client_id="client-a", environment="dev"
        )
    assert "private payload" not in str(raised.value)

    failing_upsert = Mock()
    failing_upsert.collection_exists.return_value = False
    failing_upsert.upsert.side_effect = RuntimeError("private payload")
    with pytest.raises(VectorUpsertError) as raised:
        _local_store(failing_upsert).upsert(
            [_entry()], client_id="client-a", environment="dev"
        )
    assert "private payload" not in str(raised.value)


def test_qdrant_malformed_results_are_rejected():
    client = Mock()
    client.collection_exists.return_value = True
    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=SimpleNamespace(size=2, distance="Cosine")
            )
        )
    )
    client.query_points.return_value = SimpleNamespace(
        points=[SimpleNamespace(score=float("nan"), payload={})]
    )

    with pytest.raises(VectorRetrievalError, match="malformed"):
        _local_store(client).retrieve(
            [1.0, 0.0], client_id="client-a", environment="dev"
        )


def test_qdrant_and_memory_stores_have_same_deterministic_ranking():
    entries = [
        _entry(vector=(1.0, 0.0)),
        _entry(
            document_id="document-2",
            chunk_id="document-2:000000",
            vector=(0.8, 0.2),
        ),
        _entry(
            document_id="document-3",
            chunk_id="document-3:000000",
            vector=(0.0, 1.0),
        ),
    ]
    memory = InMemoryVectorStore()
    memory.upsert(entries, client_id="client-a", environment="dev")
    qdrant = _local_store()
    qdrant.upsert(entries, client_id="client-a", environment="dev")

    memory_ids = [
        result.document_id
        for result in memory.retrieve(
            [1.0, 0.0], client_id="client-a", environment="dev", top_k=3
        )
    ]
    qdrant_ids = [
        result.document_id
        for result in qdrant.retrieve(
            [1.0, 0.0], client_id="client-a", environment="dev", top_k=3
        )
    ]

    assert qdrant_ids == memory_ids


def test_offline_evaluator_runs_qdrant_with_fake_embeddings():
    provider = DeterministicFakeEmbeddingProvider(dimensions=8)
    query = "glue access denied"
    relevant_vector = tuple(provider.embed([query])[0])
    store = _local_store()
    store.upsert(
        [
            _entry(
                document_id="relevant",
                chunk_id="relevant:000000",
                vector=relevant_vector,
            )
        ],
        client_id="client-a",
        environment="dev",
    )

    summary = RetrievalEvaluator().evaluate_vector_store(
        [
            RetrievalEvaluationCase(
                query=query,
                expected_document_ids=frozenset({"relevant"}),
            )
        ],
        embedding_provider=provider,
        vector_store=store,
        client_id="client-a",
        environment="dev",
        k=1,
        minimum_similarity=-1.0,
    )

    assert summary.recall_at_k == 1.0
    assert summary.mean_reciprocal_rank == 1.0


def test_rag_application_uses_scoped_vector_store_and_preserves_sources():
    embedding_provider = DeterministicFakeEmbeddingProvider(dimensions=8)
    embedding_provider.embed = Mock(return_value=[[1.0, 0.0]])
    vector_store = Mock()
    vector_store.provider_name = "qdrant"
    vector_store.retrieve.return_value = [
        RetrievalResult(
            document_id="document-1",
            chunk_id="document-1:000000",
            source="Runbook",
            text="Use the scoped Glue execution role as evidence.",
            similarity_score=0.9,
            metadata={
                "client_id": "client-a",
                "environment": "dev",
                "knowledge_namespace": "runbooks",
                "source_object_key": "knowledge/raw/document-1/guide.txt",
            },
        )
    ]
    routing_config = ClassificationRoutingConfig(default_retrieval_top_k=3)
    application_config = ApplicationConfig(
        maximum_retrieved_chunks=3,
        minimum_similarity=0.2,
    )
    application = RAGApplicationService(
        classifier=RuleBasedIntentClassifier(routing_config),
        router=RequestRouter(routing_config),
        embedding_provider=embedding_provider,
        retriever=None,
        vector_store=vector_store,
        prompt_builder=GroundedPromptBuilder(
            prompt_version=application_config.prompt_version
        ),
        llm_provider=DeterministicFakeLLMProvider(
            response_text="Grounded result [S1]."
        ),
        config=application_config,
        runtime_mode="demo",
    )

    response = application.handle(
        ApplicationRequest(
            request_id="scoped-vector-request",
            query="Find retry guidance in the runbook",
            client_id="client-a",
            environment="dev",
            metadata={"knowledge_namespace": "runbooks"},
            timestamp=FIXED_TIME,
        )
    )

    assert response.status == ApplicationStatus.COMPLETED
    assert response.sources[0].source_name == "Runbook"
    embedding_provider.embed.assert_called_once_with(
        ["Find retry guidance in the runbook"]
    )
    vector_store.retrieve.assert_called_once_with(
        [1.0, 0.0],
        client_id="client-a",
        environment="dev",
        filters={"knowledge_namespace": "runbooks"},
        top_k=3,
        minimum_similarity=0.2,
    )


def test_pending_ingestion_flows_through_embedding_and_vector_indexing():
    storage = MemoryKnowledgeStorage()
    pipeline = KnowledgeIngestionPipeline(
        storage,
        clock=lambda: FIXED_TIME,
        document_id_factory=lambda: "document-local",
    )
    entry = pipeline.ingest(
        filename="guide.txt",
        content=b"alpha beta gamma",
        source="manual-upload",
    )
    provider = DeterministicFakeEmbeddingProvider(dimensions=8)
    provider.embed = Mock(wraps=provider.embed)
    manifest = KnowledgeManifestRepository(storage)
    vector_store = InMemoryVectorStore(minimum_similarity=-1.0)
    workflow = VectorIndexingWorkflow(
        storage=storage,
        embedding_workflow=EmbeddingWorkflow(
            storage=storage,
            provider=provider,
            model_id=provider.model_id,
            batch_size=8,
            manifest=manifest,
            clock=lambda: FIXED_TIME,
        ),
        vector_store=vector_store,
        manifest=manifest,
    )

    first = workflow.index_pending_document(
        entry,
        client_id="Client A",
        environment="dev",
        knowledge_namespace="runbooks",
    )
    second = workflow.index_pending_document(
        entry,
        client_id="Client A",
        environment="dev",
        knowledge_namespace="runbooks",
    )

    assert first.vector_status == VectorIngestionStatus.COMPLETE
    assert first.upserted_count == entry.chunk_count
    assert second.embedding_report.created == ()
    assert second.embedding_report.skipped_chunk_ids == ()
    assert second.statistics.already_indexed_chunk_count == entry.chunk_count
    assert second.upserted_count == 0
    assert provider.embed.call_count == 1
    persisted = storage.json_objects[KnowledgeKeys.MANIFEST]["documents"][
        entry.document_id
    ]
    assert persisted["embedding_status"] == "complete"
    assert persisted["vector_status"] == "complete"
    assert persisted["vector_store_provider"] == "memory"
    results = vector_store.retrieve(
        provider.embed(["alpha"])[0],
        client_id="client-a",
        environment="dev",
        filters={"knowledge_namespace": "runbooks"},
    )
    assert results
    assert results[0].metadata["document_hash"] == entry.metadata.checksum
    assert results[0].metadata["source"] == "manual-upload"
