import json
import math
from pathlib import Path

import pytest

from evaluation.benchmark import (
    REQUIRED_CATEGORIES,
    REQUIRED_MATCH_TYPES,
    BenchmarkCase,
    load_benchmark,
)
from evaluation.comparison import (
    ComparisonConfig,
    _corpus_checksum,
    _evaluate_outcome,
    _summarize,
    run_comparison,
)
from evaluation.reporting import (
    CSV_FILENAME,
    JSON_FILENAME,
    MARKDOWN_FILENAME,
    render_csv,
    render_markdown,
    write_reports,
)
from evaluation.run_retrieval_comparison import (
    DEFAULT_BENCHMARK,
    main,
)
from knowledge.hybrid_retrieval import (
    ReciprocalRankFusionConfig,
    ReciprocalRankFusionRetriever,
)
from knowledge.keyword_retrieval import (
    BM25Config,
    InMemoryBM25Retriever,
)
from knowledge.models import EmbeddingRecord
from knowledge.retrieval import RetrievalEntry, RetrievalResult


def test_corpus_checksum_is_stable_across_line_endings(tmp_path):
    lf_corpus = tmp_path / "lf"
    crlf_corpus = tmp_path / "crlf"
    lf_corpus.mkdir()
    crlf_corpus.mkdir()
    content = "# Synthetic guide\n\nLine one.\nLine two.\n"
    (lf_corpus / "guide.md").write_bytes(content.encode("utf-8"))
    (crlf_corpus / "guide.md").write_bytes(
        content.replace("\n", "\r\n").encode("utf-8")
    )

    assert _corpus_checksum(lf_corpus) == _corpus_checksum(crlf_corpus)


def _entry(
    document_id: str,
    text: str,
    *,
    vector: tuple[float, ...] = (1.0, 0.0),
    client_id: str = "client-a",
    environment: str = "dev",
    source: str | None = None,
) -> RetrievalEntry:
    chunk_id = f"{document_id}:000000"
    return RetrievalEntry(
        embedding_record=EmbeddingRecord(
            schema_version=EmbeddingRecord.CURRENT_SCHEMA_VERSION,
            document_id=document_id,
            chunk_id=chunk_id,
            chunk_text_checksum="a" * 64,
            embedding_model_id="test-model",
            embedding_dimensions=len(vector),
            embedding_vector=vector,
            creation_timestamp="2026-07-27T00:00:00Z",
            source_object_key=f"demo://{document_id}.md",
        ),
        source=source or document_id,
        text=text,
        metadata={
            "client_id": client_id,
            "environment": environment,
            "topic": document_id,
            "custom": "preserved",
        },
    )


def _result(document_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        document_id=document_id,
        chunk_id=f"{document_id}:000000",
        source=document_id,
        text=document_id,
        similarity_score=score,
        metadata={},
    )


class _StepClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


def test_bm25_ranks_exact_terms_and_preserves_metadata():
    retriever = InMemoryBM25Retriever(
        [
            _entry("doc-a", "Glue execution role AccessDenied bucket"),
            _entry("doc-b", "Athena workgroup query output"),
        ]
    )

    results = retriever.retrieve(
        "Glue AccessDenied role",
        client_id="client-a",
        environment="dev",
    )

    assert [result.document_id for result in results] == ["doc-a"]
    assert results[0].metadata["custom"] == "preserved"
    assert results[0].source == "doc-a"


def test_bm25_ties_are_deterministic_and_top_k_is_enforced():
    retriever = InMemoryBM25Retriever(
        [
            _entry("doc-b", "same shared token"),
            _entry("doc-a", "same shared token"),
        ],
        config=BM25Config(top_k=1),
    )

    first = retriever.retrieve(
        "shared",
        client_id="client-a",
        environment="dev",
    )
    second = retriever.retrieve(
        "shared",
        client_id="client-a",
        environment="dev",
    )

    assert [result.document_id for result in first] == ["doc-a"]
    assert first == second


def test_bm25_minimum_score_and_no_result_query():
    retriever = InMemoryBM25Retriever(
        [_entry("doc-a", "known searchable phrase")]
    )

    assert retriever.retrieve(
        "absent-token",
        client_id="client-a",
        environment="dev",
    ) == []
    assert retriever.retrieve(
        "known",
        client_id="client-a",
        environment="dev",
        minimum_score=100.0,
    ) == []


def test_bm25_upsert_removes_duplicate_document_chunk_pairs():
    original = _entry("doc-a", "old phrase")
    replacement = _entry("doc-a", "new phrase", source="replacement")
    retriever = InMemoryBM25Retriever([original, replacement])

    results = retriever.retrieve(
        "new phrase",
        client_id="client-a",
        environment="dev",
    )

    assert len(results) == 1
    assert results[0].source == "replacement"


@pytest.mark.parametrize(
    ("client_id", "environment", "expected"),
    [
        ("client-a", "dev", ["client-a-dev"]),
        ("client-b", "dev", ["client-b-dev"]),
        ("client-a", "prod", ["client-a-prod"]),
    ],
)
def test_bm25_enforces_client_and_environment_isolation(
    client_id,
    environment,
    expected,
):
    retriever = InMemoryBM25Retriever(
        [
            _entry("client-a-dev", "shared", client_id="client-a"),
            _entry("client-b-dev", "shared", client_id="client-b"),
            _entry(
                "client-a-prod",
                "shared",
                client_id="client-a",
                environment="prod",
            ),
        ]
    )

    results = retriever.retrieve(
        "shared",
        client_id=client_id,
        environment=environment,
    )

    assert [result.document_id for result in results] == expected


def test_reciprocal_rank_fusion_combines_semantic_and_keyword_rankings():
    retriever = ReciprocalRankFusionRetriever(
        [
            _entry("semantic-first", "apple", vector=(1.0, 0.0)),
            _entry("keyword-first", "orange", vector=(0.8, 0.2)),
        ],
        config=ReciprocalRankFusionConfig(rank_constant=10),
    )

    results = retriever.retrieve(
        (1.0, 0.0),
        "orange",
        client_id="client-a",
        environment="dev",
    )

    assert results[0].document_id == "keyword-first"
    assert results[0].metadata["fusion_method"] == (
        "reciprocal_rank_fusion"
    )
    assert results[0].metadata["keyword_rank"] == 1
    assert results[0].metadata["semantic_rank"] == 2


def test_hybrid_removes_duplicates_enforces_top_k_and_minimum_score():
    retriever = ReciprocalRankFusionRetriever(
        [
            _entry("doc-a", "alpha", vector=(1.0, 0.0)),
            _entry("doc-a", "alpha replacement", vector=(1.0, 0.0)),
            _entry("doc-b", "beta", vector=(0.0, 1.0)),
        ]
    )

    results = retriever.retrieve(
        (1.0, 0.0),
        "alpha",
        client_id="client-a",
        environment="dev",
        top_k=1,
    )
    filtered = retriever.retrieve(
        (1.0, 0.0),
        "alpha",
        client_id="client-a",
        environment="dev",
        minimum_score=1.0,
    )

    assert len(results) == 1
    assert len({(item.document_id, item.chunk_id) for item in results}) == 1
    assert filtered == []


def test_hybrid_enforces_client_and_environment_isolation():
    retriever = ReciprocalRankFusionRetriever(
        [
            _entry("allowed", "shared term"),
            _entry("other-client", "shared term", client_id="client-b"),
            _entry(
                "other-env",
                "shared term",
                environment="prod",
            ),
        ]
    )

    results = retriever.retrieve(
        (1.0, 0.0),
        "shared",
        client_id="client-a",
        environment="dev",
    )

    assert [result.document_id for result in results] == ["allowed"]


@pytest.mark.parametrize(
    "config",
    [
        lambda: BM25Config(k1=0),
        lambda: BM25Config(b=1.1),
        lambda: ReciprocalRankFusionConfig(semantic_weight=0),
        lambda: ReciprocalRankFusionConfig(rank_constant=0),
        lambda: ReciprocalRankFusionConfig(
            candidate_pool_size=4,
            top_k=5,
        ),
        lambda: ComparisonConfig(top_k=3),
    ],
)
def test_retrieval_configuration_validation(config):
    with pytest.raises(ValueError):
        config()


def test_benchmark_is_synthetic_complete_and_reproducible():
    benchmark = load_benchmark(DEFAULT_BENCHMARK)

    assert len(benchmark.cases) == 35
    assert {case.category for case in benchmark.cases} == (
        REQUIRED_CATEGORIES
    )
    assert {case.match_type for case in benchmark.cases} == (
        REQUIRED_MATCH_TYPES
    )
    assert benchmark.license == "CC0-1.0"
    assert len({case.query_id for case in benchmark.cases}) == 35


def test_benchmark_validation_rejects_duplicate_query_ids(tmp_path):
    payload = json.loads(DEFAULT_BENCHMARK.read_text(encoding="utf-8"))
    payload["queries"][1]["query_id"] = payload["queries"][0]["query_id"]
    malformed = tmp_path / "malformed.json"
    malformed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate query_id"):
        load_benchmark(malformed)


def test_metric_calculation_handles_hits_misses_and_no_results():
    hit_case = BenchmarkCase(
        query_id="metric-hit",
        query="hit",
        expected_document_ids=("doc-b",),
        expected_chunk_ids=(),
        category="monitoring",
        difficulty="easy",
        match_type="exact_keyword",
        notes="Synthetic metric test.",
    )
    miss_case = BenchmarkCase(
        query_id="metric-miss",
        query="miss",
        expected_document_ids=("doc-c",),
        expected_chunk_ids=(),
        category="monitoring",
        difficulty="hard",
        match_type="paraphrase",
        notes="Synthetic metric test.",
    )
    hit = _evaluate_outcome(
        "keyword",
        hit_case,
        [_result("doc-a", 1.0), _result("doc-b", 0.5)],
        latency_ms=2.0,
    )
    miss = _evaluate_outcome(
        "keyword",
        miss_case,
        [],
        latency_ms=4.0,
    )

    summary = _summarize([hit, miss])

    assert hit.precision_at[1] == 0.0
    assert hit.precision_at[3] == pytest.approx(1 / 3)
    assert hit.recall_at[3] == 1.0
    assert hit.reciprocal_rank == 0.5
    assert summary["hit_rate"] == 0.5
    assert summary["no_result_rate"] == 0.5
    assert summary["average_latency_ms"] == 3.0
    assert summary["p50_latency_ms"] == 3.0
    assert summary["p95_latency_ms"] == 4.0


def test_comparison_output_is_deterministic_with_fixed_clock_and_timestamp():
    benchmark = load_benchmark(DEFAULT_BENCHMARK)
    config = ComparisonConfig(latency_repetitions=1)

    first = run_comparison(
        benchmark,
        config=config,
        evaluated_at="2026-07-27T00:00:00Z",
        clock=_StepClock(),
    )
    second = run_comparison(
        benchmark,
        config=config,
        evaluated_at="2026-07-27T00:00:00Z",
        clock=_StepClock(),
    )

    assert first.to_dict() == second.to_dict()
    assert first.selection["recommended_default"] == "keyword"
    assert first.selection["application_default_changed"] is False
    assert math.isclose(first.metrics["keyword"]["p50_latency_ms"], 1.0)


def test_report_generation_writes_json_markdown_and_csv(tmp_path):
    comparison = run_comparison(
        load_benchmark(DEFAULT_BENCHMARK),
        config=ComparisonConfig(latency_repetitions=1),
        evaluated_at="2026-07-27T00:00:00Z",
        clock=_StepClock(),
    )

    paths = write_reports(comparison, tmp_path)

    assert [path.name for path in paths] == [
        JSON_FILENAME,
        MARKDOWN_FILENAME,
        CSV_FILENAME,
    ]
    payload = json.loads((tmp_path / JSON_FILENAME).read_text())
    markdown = render_markdown(comparison)
    csv_report = render_csv(comparison)
    assert payload["selection"]["recommended_default"] == "keyword"
    assert "| semantic |" in markdown
    assert "| keyword |" in markdown
    assert "| hybrid |" in markdown
    assert csv_report.count("\n") == 106


def test_runner_returns_nonzero_for_malformed_benchmark(tmp_path):
    malformed = tmp_path / "broken.json"
    malformed.write_text("not-json", encoding="utf-8")

    assert main(
        [
            "--benchmark",
            str(malformed),
            "--output-dir",
            str(tmp_path / "results"),
        ]
    ) == 2


def test_runner_returns_nonzero_for_invalid_configuration(tmp_path):
    assert main(
        [
            "--top-k",
            "3",
            "--output-dir",
            str(tmp_path),
        ]
    ) == 2
