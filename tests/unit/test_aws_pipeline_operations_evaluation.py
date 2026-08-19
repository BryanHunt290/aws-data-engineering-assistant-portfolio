import json
from pathlib import Path

from evaluation.aws_pipeline_operations_benchmark import (
    DEFAULT_DATASET_ROOT,
    load_aws_pipeline_operations_benchmarks,
)
from evaluation.run_aws_pipeline_operations_evaluation import main
from ui.bootstrap import load_demo_documents


def test_dataset_corpus_loader_preserves_reviewed_metadata():
    documents = load_demo_documents(DEFAULT_DATASET_ROOT / "documents")
    by_id = {document.document_id: document for document in documents}

    assert len(documents) == 36
    document = by_id["apo-001-s3-prefix-isolation"]
    assert document.title == "S3 Prefix Isolation for Shared Pipeline Buckets"
    assert document.topic == "Amazon S3"
    assert document.license == "CC-BY-4.0"
    assert document.object_key == (
        "dataset://aws-pipeline-operations/s3_prefix_isolation.md"
    )


def test_test_split_adapter_accounts_for_every_evaluation_case():
    benchmarks = load_aws_pipeline_operations_benchmarks(split="test")

    assert len(benchmarks.document_ids) == 6
    assert benchmarks.retrieval_query_count == 36
    assert len(benchmarks.retrieval.cases) == 30
    assert len(benchmarks.unanswerable_query_ids) == 6
    assert len(benchmarks.answers.cases) == 18
    assert {case.match_type for case in benchmarks.retrieval.cases} == {
        "exact_keyword",
        "paraphrase",
        "ambiguous",
    }
    assert all(
        set(case.expected_document_ids).issubset(benchmarks.document_ids)
        for case in benchmarks.retrieval.cases
    )
    assert any(case.safety_sensitive for case in benchmarks.answers.cases)


def test_runner_writes_sanitized_offline_evidence(tmp_path: Path):
    exit_code = main(
        [
            "--split",
            "test",
            "--output-dir",
            str(tmp_path),
            "--latency-repetitions",
            "1",
            "--evaluated-at",
            "2026-08-06T00:00:00Z",
            "--git-commit",
            "test-commit",
        ]
    )

    assert exit_code == 0
    summary = json.loads(
        (tmp_path / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    assert summary["dataset"] == {
        "name": "AWS Data Pipeline Operations Knowledge Base",
        "version": "1.0.0",
        "split": "test",
        "license": "CC-BY-4.0",
        "document_count": 6,
        "retrieval_query_count": 36,
        "answerable_retrieval_query_count": 30,
        "unanswerable_retrieval_query_count": 6,
        "answer_case_count": 18,
    }
    assert summary["retrieval"]["recommended_strategy"] in {
        "semantic",
        "keyword",
        "hybrid",
    }
    assert summary["answers"]["recommended_prompt_strategy"]
    assert (tmp_path / "retrieval" / "retrieval_comparison.json").is_file()
    assert (tmp_path / "retrieval" / "retrieval_comparison.md").is_file()
    assert (tmp_path / "retrieval" / "retrieval_query_results.csv").is_file()
    assert (tmp_path / "answers" / "llm_prompt_comparison.json").is_file()
    assert (tmp_path / "answers" / "llm_prompt_comparison.md").is_file()
    assert (tmp_path / "answers" / "llm_prompt_case_results.csv").is_file()
