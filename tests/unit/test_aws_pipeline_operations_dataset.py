from __future__ import annotations

import json
from pathlib import Path

from knowledge.config import KnowledgeConfig
from knowledge.evaluation import RetrievalEvaluationCase
from knowledge.ingestion import KnowledgeIngestionPipeline
from knowledge.storage import FileSystemKnowledgeStorage, KnowledgeKeys
from scripts.generate_aws_pipeline_operations_dataset import (
    DATASET_ROOT,
    DOCUMENT_SPECS,
    build_records,
    check_dataset,
    render_dataset_files,
)
from scripts.validate_aws_pipeline_operations_dataset import (
    ANSWER_FIELDS,
    METADATA_FIELDS,
    RETRIEVAL_FIELDS,
    find_near_duplicate_questions,
    validate_dataset,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_dataset_has_required_schemas_counts_and_references():
    metadata, queries, answers = build_records()

    assert len(DOCUMENT_SPECS) == 36
    assert len(metadata) == 36
    assert len(queries) == 216
    assert len(answers) == 108
    assert all(METADATA_FIELDS <= set(record) for record in metadata)
    assert all(RETRIEVAL_FIELDS <= set(record) for record in queries)
    assert all(ANSWER_FIELDS <= set(record) for record in answers)

    document_ids = {record["document_id"] for record in metadata}
    assert len(document_ids) == len(metadata)
    assert all(
        set(record["relevant_document_ids"]) <= document_ids
        for record in queries
    )
    assert all(
        set(record["supporting_document_ids"]) <= document_ids
        for record in answers
    )
    assert all(
        sum(
            record["primary_document_id"] == document_id
            or document_id in record["forbidden_document_ids"]
            for record in queries
        )
        == 6
        for document_id in document_ids
    )


def test_static_dataset_matches_deterministic_generator():
    first = render_dataset_files()
    second = render_dataset_files()

    assert first == second
    assert check_dataset(REPOSITORY_ROOT) == []


def test_validator_confirms_split_isolation_and_exact_counts():
    summary = validate_dataset(REPOSITORY_ROOT)

    assert summary == {
        "documents": 36,
        "metadata_records": 36,
        "retrieval_queries": 216,
        "answer_cases": 108,
        "splits": {
            "train": {
                "documents": 25,
                "retrieval_queries": 150,
                "answer_cases": 75,
            },
            "validation": {
                "documents": 5,
                "retrieval_queries": 30,
                "answer_cases": 15,
            },
            "test": {
                "documents": 6,
                "retrieval_queries": 36,
                "answer_cases": 18,
            },
        },
        "generated_files": 42,
    }

    split_directory = REPOSITORY_ROOT / DATASET_ROOT / "splits"
    document_membership: dict[str, str] = {}
    query_membership: dict[str, str] = {}
    for name in ("train", "validation", "test"):
        split = json.loads(
            (split_directory / f"{name}.json").read_text(encoding="utf-8")
        )
        for document_id in split["document_ids"]:
            assert document_id not in document_membership
            document_membership[document_id] = name
        for query_id in split["retrieval_query_ids"]:
            assert query_id not in query_membership
            query_membership[query_id] = name

    _, queries, _ = build_records()
    for record in queries:
        document_id = record["primary_document_id"]
        if document_id is None:
            document_id = record["forbidden_document_ids"][0]
        assert query_membership[record["query_id"]] == document_membership[
            document_id
        ]


def test_near_duplicate_detector_finds_accidental_variants():
    records = [
        {
            "query_id": "first",
            "query": "How should the pipeline verify a partition before promotion?",
        },
        {
            "query_id": "second",
            "query": "How should the pipeline verify a partition before promotion!",
        },
        {
            "query_id": "different",
            "query": "What does a bounded EventBridge archive replay preserve?",
        },
    ]

    assert find_near_duplicate_questions(records) == [("first", "second")]


def test_markdown_document_uses_existing_ingestion_and_evaluation_models(tmp_path):
    metadata, queries, _ = build_records()
    document = metadata[0]
    document_path = (
        REPOSITORY_ROOT / DATASET_ROOT / "documents" / document["filename"]
    )
    storage = FileSystemKnowledgeStorage(tmp_path / "knowledge-store")
    pipeline = KnowledgeIngestionPipeline(
        storage,
        config=KnowledgeConfig(chunk_size=800, overlap=80),
        document_id_factory=lambda: document["document_id"],
    )

    entry = pipeline.ingest(
        filename=document["filename"],
        content=document_path.read_bytes(),
        source=f"dataset://{document['filename']}",
    )

    assert entry.document_id == document["document_id"]
    assert entry.metadata.file_type == "md"
    assert entry.chunk_count > 1
    chunks = storage.get_json(KnowledgeKeys.chunks(entry.document_id))
    assert chunks is not None
    assert len(chunks["chunks"]) == entry.chunk_count

    answerable = next(record for record in queries if record["answerable"])
    evaluation_case = RetrievalEvaluationCase(
        query=answerable["query"],
        expected_document_ids=frozenset(
            answerable["relevant_document_ids"]
        ),
    )
    assert evaluation_case.expected_document_ids == frozenset(
        {answerable["primary_document_id"]}
    )
