"""Build and evaluate semantic, keyword, and hybrid local retrievers."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import platform
from statistics import median
from time import perf_counter
from typing import Any, Callable, Sequence

from evaluation.benchmark import BenchmarkCase, RetrievalBenchmark
from knowledge.chunking import TextChunker
from knowledge.hybrid_retrieval import (
    ReciprocalRankFusionConfig,
    ReciprocalRankFusionRetriever,
)
from knowledge.keyword_retrieval import BM25Config, InMemoryBM25Retriever
from knowledge.models import EmbeddingRecord, KnowledgeChunk
from knowledge.retrieval import (
    InMemoryCosineRetriever,
    RetrievalEntry,
    RetrievalResult,
)
from ui.bootstrap import (
    DEMO_CORPUS_DIRECTORY,
    DEMO_CREATION_TIMESTAMP,
    DeterministicDemoEmbeddingProvider,
    DemoDocument,
    load_demo_documents,
)


APPLICATION_VERSION = "retrieval-evaluation-v1"
STRATEGIES = ("semantic", "keyword", "hybrid")
EVALUATION_K_VALUES = (1, 3, 5)


@dataclass(frozen=True)
class ComparisonConfig:
    """Validated configuration for one comparison run."""

    top_k: int = 5
    semantic_minimum_similarity: float = 0.0
    keyword_minimum_score: float = 0.0
    hybrid_minimum_score: float = 0.0
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    semantic_weight: float = 1.0
    keyword_weight: float = 1.0
    rrf_rank_constant: int = 60
    candidate_pool_size: int = 50
    latency_repetitions: int = 5

    def __post_init__(self) -> None:
        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k < max(EVALUATION_K_VALUES)
        ):
            raise ValueError("top_k must be an integer of at least 5")
        if (
            isinstance(self.latency_repetitions, bool)
            or not isinstance(self.latency_repetitions, int)
            or self.latency_repetitions <= 0
        ):
            raise ValueError("latency_repetitions must be greater than zero")
        _validate_finite_range(
            self.semantic_minimum_similarity,
            "semantic_minimum_similarity",
            minimum=-1.0,
            maximum=1.0,
        )
        for name, value in (
            ("keyword_minimum_score", self.keyword_minimum_score),
            ("hybrid_minimum_score", self.hybrid_minimum_score),
        ):
            _validate_finite_range(value, name, minimum=0.0)
        BM25Config(
            k1=self.bm25_k1,
            b=self.bm25_b,
            top_k=self.top_k,
            minimum_score=self.keyword_minimum_score,
        )
        ReciprocalRankFusionConfig(
            semantic_weight=self.semantic_weight,
            keyword_weight=self.keyword_weight,
            rank_constant=self.rrf_rank_constant,
            candidate_pool_size=self.candidate_pool_size,
            top_k=self.top_k,
            minimum_score=self.hybrid_minimum_score,
        )


@dataclass(frozen=True)
class QueryOutcome:
    """One query's ranking and measurements for one strategy."""

    strategy: str
    case: BenchmarkCase
    results: tuple[RetrievalResult, ...]
    latency_ms: float
    precision_at: dict[int, float]
    recall_at: dict[int, float]
    reciprocal_rank: float
    hit: bool
    exact_document_success: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "query_id": self.case.query_id,
            "query": self.case.query,
            "expected_document_ids": list(
                self.case.expected_document_ids
            ),
            "expected_chunk_ids": list(self.case.expected_chunk_ids),
            "category": self.case.category,
            "difficulty": self.case.difficulty,
            "match_type": self.case.match_type,
            "notes": self.case.notes,
            "latency_ms": round(self.latency_ms, 6),
            "precision_at_1": self.precision_at[1],
            "precision_at_3": self.precision_at[3],
            "precision_at_5": self.precision_at[5],
            "recall_at_1": self.recall_at[1],
            "recall_at_3": self.recall_at[3],
            "recall_at_5": self.recall_at[5],
            "reciprocal_rank": self.reciprocal_rank,
            "hit": self.hit,
            "exact_document_success": self.exact_document_success,
            "results": [
                {
                    "rank": rank,
                    "document_id": result.document_id,
                    "chunk_id": result.chunk_id,
                    "source": result.source,
                    "score": round(result.similarity_score, 10),
                }
                for rank, result in enumerate(self.results, start=1)
            ],
        }


@dataclass(frozen=True)
class RetrievalComparison:
    """Complete comparison payload and reviewer-facing selection."""

    metadata: dict[str, Any]
    settings: dict[str, Any]
    metrics: dict[str, dict[str, Any]]
    selection: dict[str, Any]
    failure_analysis: dict[str, dict[str, list[str]]]
    outcomes: tuple[QueryOutcome, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "metadata": self.metadata,
            "settings": self.settings,
            "metrics": self.metrics,
            "selection": self.selection,
            "failure_analysis": self.failure_analysis,
            "query_results": [
                outcome.to_dict() for outcome in self.outcomes
            ],
        }


def run_comparison(
    benchmark: RetrievalBenchmark,
    *,
    config: ComparisonConfig | None = None,
    corpus_directory: Path | None = None,
    evaluated_at: str | None = None,
    clock: Callable[[], float] = perf_counter,
) -> RetrievalComparison:
    """Run all strategies against the same validated corpus and cases."""

    settings = config or ComparisonConfig()
    timestamp = evaluated_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    _validate_evaluation_timestamp(timestamp)
    corpus_path = corpus_directory or DEMO_CORPUS_DIRECTORY
    provider = DeterministicDemoEmbeddingProvider()
    documents = load_demo_documents(corpus_path)
    entries = _build_entries(
        documents,
        provider,
        client_id=benchmark.client_id,
        environment=benchmark.environment,
    )
    _validate_benchmark_targets(benchmark, entries)

    semantic = InMemoryCosineRetriever(
        entries,
        top_k=settings.top_k,
        minimum_similarity=settings.semantic_minimum_similarity,
    )
    keyword = InMemoryBM25Retriever(
        entries,
        config=BM25Config(
            k1=settings.bm25_k1,
            b=settings.bm25_b,
            top_k=settings.top_k,
            minimum_score=settings.keyword_minimum_score,
        ),
    )
    hybrid = ReciprocalRankFusionRetriever(
        entries,
        config=ReciprocalRankFusionConfig(
            semantic_weight=settings.semantic_weight,
            keyword_weight=settings.keyword_weight,
            rank_constant=settings.rrf_rank_constant,
            candidate_pool_size=settings.candidate_pool_size,
            top_k=settings.top_k,
            minimum_score=settings.hybrid_minimum_score,
        ),
        keyword_config=BM25Config(
            k1=settings.bm25_k1,
            b=settings.bm25_b,
            top_k=settings.candidate_pool_size,
            minimum_score=0.0,
        ),
    )

    outcomes: list[QueryOutcome] = []
    for case in benchmark.cases:
        calls: dict[str, Callable[[], list[RetrievalResult]]] = {
            "semantic": lambda case=case: semantic.retrieve(
                provider.embed([case.query])[0],
                top_k=settings.top_k,
                minimum_similarity=settings.semantic_minimum_similarity,
            ),
            "keyword": lambda case=case: keyword.retrieve(
                case.query,
                client_id=benchmark.client_id,
                environment=benchmark.environment,
                top_k=settings.top_k,
                minimum_score=settings.keyword_minimum_score,
            ),
            "hybrid": lambda case=case: hybrid.retrieve(
                provider.embed([case.query])[0],
                case.query,
                client_id=benchmark.client_id,
                environment=benchmark.environment,
                top_k=settings.top_k,
                minimum_score=settings.hybrid_minimum_score,
            ),
        }
        for strategy in STRATEGIES:
            results, latency_ms = _measure(
                calls[strategy],
                repetitions=settings.latency_repetitions,
                clock=clock,
            )
            outcomes.append(
                _evaluate_outcome(
                    strategy,
                    case,
                    results,
                    latency_ms=latency_ms,
                )
            )

    outcomes_tuple = tuple(outcomes)
    metrics = {
        strategy: _summarize(
            [
                outcome
                for outcome in outcomes_tuple
                if outcome.strategy == strategy
            ]
        )
        for strategy in STRATEGIES
    }
    selection = _select_strategies(metrics)
    failures = {
        strategy: {
            "missed_at_5": [
                outcome.case.query_id
                for outcome in outcomes_tuple
                if outcome.strategy == strategy and not outcome.hit
            ],
            "no_results": [
                outcome.case.query_id
                for outcome in outcomes_tuple
                if outcome.strategy == strategy and not outcome.results
            ],
        }
        for strategy in STRATEGIES
    }
    metadata = {
        "evaluation_date": timestamp,
        "application_version": APPLICATION_VERSION,
        "embedding_provider": provider.provider_name,
        "embedding_model_id": provider.model_id,
        "corpus_version": benchmark.corpus_version,
        "corpus_checksum": _corpus_checksum(corpus_path),
        "corpus_document_count": len(documents),
        "corpus_chunk_count": len(entries),
        "benchmark_version": benchmark.benchmark_version,
        "benchmark_case_count": len(benchmark.cases),
        "benchmark_license": benchmark.license,
        "python_version": platform.python_version(),
    }
    return RetrievalComparison(
        metadata=metadata,
        settings={
            "scope": {
                "client_id": benchmark.client_id,
                "environment": benchmark.environment,
            },
            "top_k": settings.top_k,
            "evaluated_k_values": list(EVALUATION_K_VALUES),
            "semantic_minimum_similarity": (
                settings.semantic_minimum_similarity
            ),
            "keyword_minimum_score": settings.keyword_minimum_score,
            "hybrid_minimum_score": settings.hybrid_minimum_score,
            "bm25": {
                "k1": settings.bm25_k1,
                "b": settings.bm25_b,
            },
            "fusion": {
                "method": "reciprocal_rank_fusion",
                "semantic_weight": settings.semantic_weight,
                "keyword_weight": settings.keyword_weight,
                "rank_constant": settings.rrf_rank_constant,
                "candidate_pool_size": settings.candidate_pool_size,
            },
            "latency_repetitions": settings.latency_repetitions,
        },
        metrics=metrics,
        selection=selection,
        failure_analysis=failures,
        outcomes=outcomes_tuple,
    )


def _build_entries(
    documents: Sequence[DemoDocument],
    provider: DeterministicDemoEmbeddingProvider,
    *,
    client_id: str,
    environment: str,
) -> list[RetrievalEntry]:
    chunker = TextChunker(chunk_size=1_200, overlap=150)
    chunks: list[tuple[DemoDocument, KnowledgeChunk]] = []
    for document in documents:
        chunks.extend(
            (document, chunk)
            for chunk in chunker.chunk(document.document_id, document.text)
        )
    vectors = provider.embed([chunk.text for _, chunk in chunks])
    entries: list[RetrievalEntry] = []
    for (document, chunk), vector in zip(chunks, vectors, strict=True):
        checksum = hashlib.sha256(
            chunk.text.encode("utf-8")
        ).hexdigest()
        entries.append(
            RetrievalEntry(
                embedding_record=EmbeddingRecord(
                    schema_version=EmbeddingRecord.CURRENT_SCHEMA_VERSION,
                    document_id=document.document_id,
                    chunk_id=chunk.chunk_id,
                    chunk_text_checksum=checksum,
                    embedding_model_id=provider.model_id,
                    embedding_dimensions=len(vector),
                    embedding_vector=tuple(vector),
                    creation_timestamp=DEMO_CREATION_TIMESTAMP,
                    source_object_key=document.object_key,
                ),
                source=document.title,
                text=chunk.text,
                metadata={
                    "client_id": client_id,
                    "environment": environment,
                    "object_key": document.object_key,
                    "section": document.title,
                    "topic": document.topic,
                    "file_type": "markdown",
                    "synthetic": True,
                    "license": document.license,
                },
            )
        )
    return entries


def _validate_benchmark_targets(
    benchmark: RetrievalBenchmark,
    entries: Sequence[RetrievalEntry],
) -> None:
    document_ids = {
        entry.embedding_record.document_id for entry in entries
    }
    chunk_ids = {entry.embedding_record.chunk_id for entry in entries}
    for case in benchmark.cases:
        missing_documents = (
            set(case.expected_document_ids) - document_ids
        )
        missing_chunks = set(case.expected_chunk_ids) - chunk_ids
        if missing_documents or missing_chunks:
            missing = sorted(missing_documents | missing_chunks)
            raise ValueError(
                f"Benchmark query {case.query_id} references missing targets: "
                + ", ".join(missing)
            )


def _measure(
    operation: Callable[[], list[RetrievalResult]],
    *,
    repetitions: int,
    clock: Callable[[], float],
) -> tuple[list[RetrievalResult], float]:
    durations: list[float] = []
    reference_results: list[RetrievalResult] | None = None
    reference_ranking: tuple[tuple[str, str], ...] | None = None
    for _ in range(repetitions):
        started = clock()
        results = operation()
        elapsed_ms = max(0.0, (clock() - started) * 1_000.0)
        ranking = tuple(
            (result.document_id, result.chunk_id) for result in results
        )
        if reference_results is None:
            reference_results = results
            reference_ranking = ranking
        elif ranking != reference_ranking:
            raise ValueError("Retriever returned a nondeterministic ranking")
        durations.append(elapsed_ms)
    return reference_results or [], median(durations)


def _evaluate_outcome(
    strategy: str,
    case: BenchmarkCase,
    results: Sequence[RetrievalResult],
    *,
    latency_ms: float,
) -> QueryOutcome:
    precision_at: dict[int, float] = {}
    recall_at: dict[int, float] = {}
    expected_count = (
        len(case.expected_document_ids) + len(case.expected_chunk_ids)
    )
    for k in EVALUATION_K_VALUES:
        relevant_results = 0
        matched_targets: set[tuple[str, str]] = set()
        for result in results[:k]:
            result_relevant = False
            if result.document_id in case.expected_document_ids:
                matched_targets.add(("document", result.document_id))
                result_relevant = True
            if result.chunk_id in case.expected_chunk_ids:
                matched_targets.add(("chunk", result.chunk_id))
                result_relevant = True
            if result_relevant:
                relevant_results += 1
        precision_at[k] = relevant_results / k
        recall_at[k] = len(matched_targets) / expected_count

    reciprocal_rank = 0.0
    for rank, result in enumerate(results, start=1):
        if (
            result.document_id in case.expected_document_ids
            or result.chunk_id in case.expected_chunk_ids
        ):
            reciprocal_rank = 1.0 / rank
            break
    returned_documents = {result.document_id for result in results}
    exact_document_success = set(
        case.expected_document_ids
    ).issubset(returned_documents)
    return QueryOutcome(
        strategy=strategy,
        case=case,
        results=tuple(results),
        latency_ms=latency_ms,
        precision_at=precision_at,
        recall_at=recall_at,
        reciprocal_rank=reciprocal_rank,
        hit=reciprocal_rank > 0.0,
        exact_document_success=exact_document_success,
    )


def _summarize(outcomes: Sequence[QueryOutcome]) -> dict[str, Any]:
    if not outcomes:
        raise ValueError("Cannot summarize an empty outcome set")
    count = len(outcomes)
    latencies = sorted(outcome.latency_ms for outcome in outcomes)
    summary: dict[str, Any] = {
        "case_count": count,
        "precision_at_1": _mean(
            outcome.precision_at[1] for outcome in outcomes
        ),
        "precision_at_3": _mean(
            outcome.precision_at[3] for outcome in outcomes
        ),
        "precision_at_5": _mean(
            outcome.precision_at[5] for outcome in outcomes
        ),
        "recall_at_1": _mean(
            outcome.recall_at[1] for outcome in outcomes
        ),
        "recall_at_3": _mean(
            outcome.recall_at[3] for outcome in outcomes
        ),
        "recall_at_5": _mean(
            outcome.recall_at[5] for outcome in outcomes
        ),
        "mrr": _mean(
            outcome.reciprocal_rank for outcome in outcomes
        ),
        "hit_rate": _mean(1.0 if outcome.hit else 0.0 for outcome in outcomes),
        "no_result_rate": _mean(
            1.0 if not outcome.results else 0.0
            for outcome in outcomes
        ),
        "exact_document_success": _mean(
            1.0 if outcome.exact_document_success else 0.0
            for outcome in outcomes
        ),
        "average_latency_ms": _mean(latencies),
        "p50_latency_ms": median(latencies),
        "p95_latency_ms": _nearest_rank_percentile(latencies, 0.95),
        "by_category": _group_performance(
            outcomes,
            attribute="category",
        ),
        "by_difficulty": _group_performance(
            outcomes,
            attribute="difficulty",
        ),
        "by_match_type": _group_performance(
            outcomes,
            attribute="match_type",
        ),
    }
    summary["selection_score"] = (
        0.5 * summary["mrr"]
        + 0.3 * summary["hit_rate"]
        + 0.2 * summary["recall_at_3"]
    )
    return _round_metrics(summary)


def _group_performance(
    outcomes: Sequence[QueryOutcome],
    *,
    attribute: str,
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[QueryOutcome]] = {}
    for outcome in outcomes:
        key = str(getattr(outcome.case, attribute))
        groups.setdefault(key, []).append(outcome)
    return {
        key: {
            "case_count": len(items),
            "hit_rate": round(
                _mean(1.0 if item.hit else 0.0 for item in items),
                6,
            ),
            "mrr": round(
                _mean(item.reciprocal_rank for item in items),
                6,
            ),
            "recall_at_5": round(
                _mean(item.recall_at[5] for item in items),
                6,
            ),
        }
        for key, items in sorted(groups.items())
    }


def _select_strategies(
    metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    overall = _best_strategy(
        metrics,
        lambda values: (
            values["selection_score"],
            values["mrr"],
            values["recall_at_1"],
            -values["no_result_rate"],
        ),
    )
    exact = _best_strategy(
        metrics,
        lambda values: (
            values["by_match_type"]["exact_keyword"]["mrr"],
            values["by_match_type"]["exact_keyword"]["hit_rate"],
        ),
    )
    paraphrase = _best_strategy(
        metrics,
        lambda values: (
            values["by_match_type"]["paraphrase"]["mrr"],
            values["by_match_type"]["paraphrase"]["hit_rate"],
        ),
    )
    return {
        "recommended_default": overall,
        "best_exact_keyword": exact,
        "best_paraphrase": paraphrase,
        "selection_formula": (
            "0.5 * MRR + 0.3 * hit_rate + 0.2 * recall@3; "
            "then MRR, recall@1, and no-result rate"
        ),
        "tie_policy": (
            "Exact metric ties preserve the existing semantic default, "
            "then keyword, then hybrid."
        ),
        "application_default_changed": False,
        "backward_compatibility": (
            "The result is a documented recommendation only; the Streamlit "
            "runtime remains on its existing semantic retriever."
        ),
    }


def _best_strategy(
    metrics: dict[str, dict[str, Any]],
    score: Callable[[dict[str, Any]], tuple[float, ...]],
) -> str:
    best = STRATEGIES[0]
    best_score = score(metrics[best])
    for strategy in STRATEGIES[1:]:
        strategy_score = score(metrics[strategy])
        if strategy_score > best_score:
            best = strategy
            best_score = strategy_score
    return best


def _corpus_checksum(corpus_directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(corpus_directory.glob("*.md")):
        if path.name.casefold() == "readme.md":
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        normalized_text = (
            path.read_text(encoding="utf-8")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )
        digest.update(normalized_text.encode("utf-8"))
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _mean(values) -> float:
    materialized = tuple(float(value) for value in values)
    return sum(materialized) / len(materialized)


def _nearest_rank_percentile(
    sorted_values: Sequence[float],
    percentile: float,
) -> float:
    index = max(0, math.ceil(percentile * len(sorted_values)) - 1)
    return sorted_values[index]


def _round_metrics(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {
            key: _round_metrics(item)
            for key, item in value.items()
        }
    return value


def _validate_finite_range(
    value: float,
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
        or (maximum is not None and float(value) > maximum)
    ):
        upper = f" and {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be between {minimum}{upper}")


def _validate_evaluation_timestamp(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("evaluated_at must be a non-empty ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            "evaluated_at must be a valid ISO timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("evaluated_at must include a timezone")
