from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
import json
import logging
import math
from typing import Any, Mapping
from unittest.mock import Mock

import pytest

from knowledge.bedrock_embeddings import BedrockEmbeddingProvider
from knowledge.config import EmbeddingRetrievalConfig
from knowledge.embedding_errors import (
    EmbeddingAccessDeniedError,
    EmbeddingInvocationError,
    EmbeddingModelUnavailableError,
    EmbeddingThrottledError,
    MalformedEmbeddingResponseError,
)
from knowledge.embedding_workflow import EmbeddingWorkflow
from knowledge.evaluation import (
    RetrievalEvaluationCase,
    RetrievalEvaluator,
)
from knowledge.fake_embeddings import DeterministicFakeEmbeddingProvider
from knowledge.models import EmbeddingRecord, KnowledgeChunk
from knowledge.retrieval import (
    InMemoryCosineRetriever,
    RetrievalEntry,
    RetrievalResult,
)
from knowledge.storage import KnowledgeKeys


class MemoryKnowledgeStorage:
    def __init__(self) -> None:
        self.json_objects: dict[str, dict[str, Any]] = {}

    def put_bytes(
        self,
        key: str,
        content: bytes,
        *,
        content_type: str,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        raise AssertionError("Embedding workflow should not write raw bytes")

    def put_json(self, key: str, payload: Mapping[str, Any]) -> None:
        self.json_objects[key] = deepcopy(dict(payload))

    def get_json(self, key: str) -> dict[str, Any] | None:
        payload = self.json_objects.get(key)
        return deepcopy(payload) if payload is not None else None


class FakeBedrockError(Exception):
    def __init__(self, code: str) -> None:
        self.response = {"Error": {"Code": code}}
        super().__init__(code)


FIXED_TIME = datetime(2026, 7, 27, 14, 0, tzinfo=timezone.utc)


def _chunk(index: int, text: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=f"document-1:{index:06d}",
        document_id="document-1",
        index=index,
        text=text,
        start_character=index * 10,
        end_character=(index * 10) + len(text),
    )


def _record(
    *,
    document_id: str,
    chunk_id: str,
    vector: tuple[float, ...],
    model_id: str = "fake-v1",
) -> EmbeddingRecord:
    return EmbeddingRecord(
        schema_version=EmbeddingRecord.CURRENT_SCHEMA_VERSION,
        document_id=document_id,
        chunk_id=chunk_id,
        chunk_text_checksum="checksum",
        embedding_model_id=model_id,
        embedding_dimensions=len(vector),
        embedding_vector=vector,
        creation_timestamp="2026-07-27T14:00:00Z",
        source_object_key=f"knowledge/raw/{document_id}/source.txt",
    )


def _workflow(
    storage: MemoryKnowledgeStorage,
    provider,
    *,
    model_id: str = "fake-v1",
    batch_size: int = 2,
    event_logger: logging.Logger | None = None,
) -> EmbeddingWorkflow:
    return EmbeddingWorkflow(
        storage=storage,
        provider=provider,
        model_id=model_id,
        batch_size=batch_size,
        clock=lambda: FIXED_TIME,
        event_logger=event_logger,
    )


def test_embedding_retrieval_config_defaults_and_normalization():
    config = EmbeddingRetrievalConfig(bedrock_region=" US-WEST-2 ")

    assert config.bedrock_region == "us-west-2"
    assert config.embedding_model_id == "amazon.titan-embed-text-v2:0"
    assert config.embedding_batch_size == 8
    assert config.top_k == 5
    assert config.minimum_similarity_threshold == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bedrock_region": "invalid"}, "bedrock_region"),
        ({"embedding_model_id": " "}, "embedding_model_id"),
        ({"embedding_batch_size": 0}, "embedding_batch_size"),
        ({"top_k": 0}, "top_k"),
        (
            {"minimum_similarity_threshold": 1.1},
            "minimum_similarity_threshold",
        ),
    ],
)
def test_embedding_retrieval_config_rejects_invalid_values(
    kwargs,
    message,
):
    with pytest.raises(ValueError, match=message):
        EmbeddingRetrievalConfig(**kwargs)


def test_bedrock_provider_invokes_runtime_with_configured_model():
    client = Mock()
    client.invoke_model.return_value = {
        "body": BytesIO(b'{"embedding": [0.1, 0.2, 0.3]}')
    }
    provider = BedrockEmbeddingProvider(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="us-west-2",
        bedrock_runtime_client=client,
        dimensions=3,
    )

    vectors = provider.embed(["safe test input"])

    assert vectors == [[0.1, 0.2, 0.3]]
    request = client.invoke_model.call_args.kwargs
    assert request["modelId"] == "amazon.titan-embed-text-v2:0"
    assert request["contentType"] == "application/json"
    assert request["accept"] == "application/json"
    assert json.loads(request["body"]) == {
        "inputText": "safe test input",
        "normalize": True,
        "dimensions": 3,
    }


@pytest.mark.parametrize(
    ("code", "expected_error"),
    [
        ("ThrottlingException", EmbeddingThrottledError),
        ("AccessDeniedException", EmbeddingAccessDeniedError),
        ("ResourceNotFoundException", EmbeddingModelUnavailableError),
        ("ModelNotReadyException", EmbeddingModelUnavailableError),
        ("ServiceUnavailableException", EmbeddingModelUnavailableError),
        ("ValidationException", EmbeddingInvocationError),
    ],
)
def test_bedrock_provider_translates_service_errors(code, expected_error):
    client = Mock()
    client.invoke_model.side_effect = FakeBedrockError(code)
    provider = BedrockEmbeddingProvider(
        model_id="model-id",
        region_name="us-west-2",
        bedrock_runtime_client=client,
    )

    with pytest.raises(expected_error):
        provider.embed(["test"])


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b"{}",
        b'{"embedding": []}',
        b'{"embedding": [true, 0.2]}',
        b'{"embedding": ["0.1", 0.2]}',
        b'{"embedding": [NaN]}',
    ],
)
def test_bedrock_provider_rejects_malformed_responses(body):
    client = Mock()
    client.invoke_model.return_value = {"body": BytesIO(body)}
    provider = BedrockEmbeddingProvider(
        model_id="model-id",
        region_name="us-west-2",
        bedrock_runtime_client=client,
    )

    with pytest.raises(MalformedEmbeddingResponseError):
        provider.embed(["test"])


def test_fake_provider_is_stable_normalized_and_model_sensitive():
    first_provider = DeterministicFakeEmbeddingProvider(
        model_id="fake-v1",
        dimensions=8,
    )
    second_provider = DeterministicFakeEmbeddingProvider(
        model_id="fake-v2",
        dimensions=8,
    )

    first = first_provider.embed(["same input"])[0]
    repeated = first_provider.embed(["same input"])[0]
    different_model = second_provider.embed(["same input"])[0]

    assert first == repeated
    assert first != different_model
    assert len(first) == 8
    assert math.isclose(
        math.sqrt(sum(value * value for value in first)),
        1.0,
    )


def test_embedding_record_schema_round_trips_and_uses_versioned_key():
    record = _record(
        document_id="document-1",
        chunk_id="document-1:000000",
        vector=(0.1, 0.2),
    )

    restored = EmbeddingRecord.from_dict(record.to_dict())

    assert restored == record
    assert record.to_dict()["schema_version"] == 1
    assert KnowledgeKeys.embedding_record(
        "document-1",
        "document-1:000000",
    ) == (
        "knowledge/embeddings/document-1/"
        "document-1%3A000000.json"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 99},
        {"embedding_dimensions": 3},
        {"embedding_vector": [0.1, True]},
        {"embedding_vector": [0.1, float("inf")]},
    ],
)
def test_embedding_record_rejects_malformed_or_unknown_schema(change):
    payload = _record(
        document_id="document-1",
        chunk_id="document-1:000000",
        vector=(0.1, 0.2),
    ).to_dict()
    payload.update(change)

    with pytest.raises(ValueError):
        EmbeddingRecord.from_dict(payload)


def test_embedding_workflow_creates_records_then_skips_unchanged_chunks():
    storage = MemoryKnowledgeStorage()
    provider = DeterministicFakeEmbeddingProvider(
        model_id="fake-v1",
        dimensions=4,
    )
    provider.embed = Mock(wraps=provider.embed)
    workflow = _workflow(storage, provider)
    chunks = [_chunk(0, "alpha"), _chunk(1, "beta")]

    first_report = workflow.embed_document(
        document_id="document-1",
        chunks=chunks,
        source_object_key="knowledge/raw/document-1/source.txt",
        object_classification="indexable_text_document",
    )
    second_report = workflow.embed_document(
        document_id="document-1",
        chunks=chunks,
        source_object_key="knowledge/raw/document-1/source.txt",
        object_classification="indexable_text_document",
    )

    assert len(first_report.created) == 2
    assert first_report.failures == ()
    assert second_report.created == ()
    assert second_report.skipped_chunk_ids == (
        "document-1:000000",
        "document-1:000001",
    )
    provider.embed.assert_called_once_with(["alpha", "beta"])
    payload = storage.json_objects[
        KnowledgeKeys.embedding_record(
            "document-1",
            "document-1:000000",
        )
    ]
    assert payload["chunk_text_checksum"] == (
        "8ed3f6ad685b959ead7022518e1af76c"
        "d816f8e8ec7ccdda1ed4018e8f2223f8"
    )
    assert payload["embedding_model_id"] == "fake-v1"
    assert payload["embedding_dimensions"] == 4
    assert payload["creation_timestamp"] == "2026-07-27T14:00:00Z"
    assert payload["source_object_key"] == (
        "knowledge/raw/document-1/source.txt"
    )


def test_embedding_workflow_reembeds_changed_text_and_model():
    storage = MemoryKnowledgeStorage()
    first_provider = DeterministicFakeEmbeddingProvider(
        model_id="fake-v1",
        dimensions=4,
    )
    _workflow(storage, first_provider).embed_document(
        document_id="document-1",
        chunks=[_chunk(0, "alpha")],
        source_object_key="knowledge/raw/document-1/source.txt",
        object_classification="indexable_text_document",
    )

    changed_text_report = _workflow(
        storage,
        first_provider,
    ).embed_document(
        document_id="document-1",
        chunks=[_chunk(0, "alpha changed")],
        source_object_key="knowledge/raw/document-1/source.txt",
        object_classification="indexable_text_document",
    )
    second_provider = DeterministicFakeEmbeddingProvider(
        model_id="fake-v2",
        dimensions=4,
    )
    changed_model_report = _workflow(
        storage,
        second_provider,
        model_id="fake-v2",
    ).embed_document(
        document_id="document-1",
        chunks=[_chunk(0, "alpha changed")],
        source_object_key="knowledge/raw/document-1/source.txt",
        object_classification="indexable_text_document",
    )

    assert len(changed_text_report.created) == 1
    assert len(changed_model_report.created) == 1
    assert changed_model_report.created[0].embedding_model_id == "fake-v2"


def test_embedding_workflow_reports_partial_failures_and_continues():
    storage = MemoryKnowledgeStorage()
    provider = DeterministicFakeEmbeddingProvider(
        dimensions=4,
        fail_on_texts=frozenset({"bad"}),
    )

    report = _workflow(
        storage,
        provider,
        model_id="fake-embedding-v1",
        batch_size=1,
    ).embed_document(
        document_id="document-1",
        chunks=[
            _chunk(0, "good"),
            _chunk(1, "bad"),
            _chunk(2, "also good"),
        ],
        source_object_key="knowledge/raw/document-1/source.txt",
        object_classification="indexable_text_document",
    )

    assert [record.chunk_id for record in report.created] == [
        "document-1:000000",
        "document-1:000002",
    ]
    assert [failure.chunk_id for failure in report.failures] == [
        "document-1:000001"
    ]
    assert report.succeeded is False


def test_embedding_logs_required_fields_without_text_or_vectors(caplog):
    event_logger = logging.getLogger("test.embedding.workflow")
    caplog.set_level(logging.INFO, logger=event_logger.name)
    storage = MemoryKnowledgeStorage()
    secret_text = "sensitive-content-that-must-not-be-logged"

    _workflow(
        storage,
        DeterministicFakeEmbeddingProvider(dimensions=4),
        model_id="fake-embedding-v1",
        batch_size=1,
        event_logger=event_logger,
    ).embed_document(
        document_id="document-1",
        chunks=[_chunk(0, secret_text)],
        source_object_key="knowledge/raw/document-1/source.txt",
        object_classification="indexable_text_document",
    )

    event = json.loads(caplog.records[0].message)
    assert event["document_id"] == "document-1"
    assert event["chunk_id"] == "document-1:000000"
    assert event["model_id"] == "fake-embedding-v1"
    assert event["outcome"] == "created"
    assert event["elapsed_ms"] >= 0
    assert secret_text not in caplog.text
    assert "embedding_vector" not in caplog.text


def test_in_memory_retriever_ranks_filters_and_returns_required_fields():
    entries = [
        RetrievalEntry(
            embedding_record=_record(
                document_id="doc-a",
                chunk_id="chunk-a",
                vector=(1.0, 0.0),
            ),
            source="runbook",
            text="best match",
            metadata={"team": "data"},
        ),
        RetrievalEntry(
            embedding_record=_record(
                document_id="doc-b",
                chunk_id="chunk-b",
                vector=(0.8, 0.6),
            ),
            source="guide",
            text="second match",
            metadata={"team": "platform"},
        ),
        RetrievalEntry(
            embedding_record=_record(
                document_id="doc-c",
                chunk_id="chunk-c",
                vector=(-1.0, 0.0),
            ),
            source="archive",
            text="opposite",
            metadata={},
        ),
    ]
    retriever = InMemoryCosineRetriever(
        entries,
        top_k=2,
        minimum_similarity=0.5,
    )

    results = retriever.retrieve((1.0, 0.0))

    assert [result.chunk_id for result in results] == [
        "chunk-a",
        "chunk-b",
    ]
    assert results[0] == RetrievalResult(
        document_id="doc-a",
        chunk_id="chunk-a",
        source="runbook",
        text="best match",
        similarity_score=1.0,
        metadata={"team": "data"},
    )
    assert math.isclose(results[1].similarity_score, 0.8)


def test_in_memory_retriever_validates_query_dimensions():
    retriever = InMemoryCosineRetriever(
        [
            RetrievalEntry(
                embedding_record=_record(
                    document_id="doc-a",
                    chunk_id="chunk-a",
                    vector=(1.0, 0.0),
                ),
                source="test",
                text="test",
                metadata={},
            )
        ]
    )

    with pytest.raises(ValueError, match="dimensions"):
        retriever.retrieve((1.0, 0.0, 0.0))


def test_retrieval_evaluation_computes_representative_metrics():
    cases = [
        RetrievalEvaluationCase(
            query="where is the runbook",
            expected_document_ids=frozenset({"doc-a"}),
        ),
        RetrievalEvaluationCase(
            query="how does ingestion work",
            expected_chunk_ids=frozenset({"chunk-b"}),
        ),
    ]
    result_map = {
        "where is the runbook": [
            RetrievalResult(
                "wrong",
                "wrong-chunk",
                "test",
                "wrong",
                0.9,
                {},
            ),
            RetrievalResult(
                "doc-a",
                "chunk-a",
                "test",
                "right",
                0.8,
                {},
            ),
        ],
        "how does ingestion work": [
            RetrievalResult(
                "doc-b",
                "chunk-b",
                "test",
                "right",
                0.9,
                {},
            ),
            RetrievalResult(
                "wrong",
                "wrong-chunk",
                "test",
                "wrong",
                0.5,
                {},
            ),
        ],
    }

    summary = RetrievalEvaluator().evaluate(
        cases,
        retrieve=lambda query, k: result_map[query][:k],
        k=2,
    )

    assert summary.precision_at_k == 0.5
    assert summary.recall_at_k == 1.0
    assert summary.mean_reciprocal_rank == 0.75
    assert [case.reciprocal_rank for case in summary.cases] == [0.5, 1.0]
