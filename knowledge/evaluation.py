"""Small retrieval evaluation foundation for synthetic and curated cases."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from knowledge.embeddings import EmbeddingProvider
from knowledge.retrieval import RetrievalResult
from knowledge.vector_store import VectorStore


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    """One query and its expected relevant documents or chunks."""

    query: str
    expected_document_ids: frozenset[str] = frozenset()
    expected_chunk_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("Evaluation query cannot be empty")
        if not self.expected_document_ids and not self.expected_chunk_ids:
            raise ValueError(
                "Evaluation case must include an expected document or chunk"
            )


@dataclass(frozen=True)
class RetrievalCaseMetrics:
    """Precision, recall, and reciprocal rank for one query."""

    query: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float


@dataclass(frozen=True)
class RetrievalEvaluationSummary:
    """Mean metrics across a retrieval evaluation set."""

    precision_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float
    cases: tuple[RetrievalCaseMetrics, ...]


class RetrievalEvaluator:
    """Evaluate any retrieval implementation through a query callback."""

    def evaluate(
        self,
        cases: Sequence[RetrievalEvaluationCase],
        *,
        retrieve: Callable[[str, int], Sequence[RetrievalResult]],
        k: int,
    ) -> RetrievalEvaluationSummary:
        if not cases:
            raise ValueError("At least one evaluation case is required")
        if k <= 0:
            raise ValueError("k must be greater than zero")

        metrics = tuple(
            self.evaluate_case(
                case,
                list(retrieve(case.query, k))[:k],
                k=k,
            )
            for case in cases
        )
        count = len(metrics)
        return RetrievalEvaluationSummary(
            precision_at_k=(
                sum(item.precision_at_k for item in metrics) / count
            ),
            recall_at_k=(
                sum(item.recall_at_k for item in metrics) / count
            ),
            mean_reciprocal_rank=(
                sum(item.reciprocal_rank for item in metrics) / count
            ),
            cases=metrics,
        )

    def evaluate_vector_store(
        self,
        cases: Sequence[RetrievalEvaluationCase],
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        client_id: str,
        environment: str,
        filters: Mapping[str, Any] | None = None,
        k: int,
        minimum_similarity: float | None = None,
    ) -> RetrievalEvaluationSummary:
        """Evaluate any scoped vector store with an injected offline embedder."""

        def retrieve(query: str, limit: int) -> Sequence[RetrievalResult]:
            vectors = embedding_provider.embed([query])
            if len(vectors) != 1 or not vectors[0]:
                raise ValueError(
                    "Embedding provider returned invalid evaluation output"
                )
            return vector_store.retrieve(
                vectors[0],
                client_id=client_id,
                environment=environment,
                filters=filters,
                top_k=limit,
                minimum_similarity=minimum_similarity,
            )

        return self.evaluate(cases, retrieve=retrieve, k=k)

    def evaluate_case(
        self,
        case: RetrievalEvaluationCase,
        results: Sequence[RetrievalResult],
        *,
        k: int,
    ) -> RetrievalCaseMetrics:
        if k <= 0:
            raise ValueError("k must be greater than zero")

        relevant_results = 0
        matched_targets: set[tuple[str, str]] = set()
        reciprocal_rank = 0.0
        for rank, result in enumerate(results[:k], start=1):
            result_is_relevant = False
            if result.document_id in case.expected_document_ids:
                matched_targets.add(("document", result.document_id))
                result_is_relevant = True
            if result.chunk_id in case.expected_chunk_ids:
                matched_targets.add(("chunk", result.chunk_id))
                result_is_relevant = True
            if result_is_relevant:
                relevant_results += 1
                if reciprocal_rank == 0.0:
                    reciprocal_rank = 1.0 / rank

        expected_count = (
            len(case.expected_document_ids)
            + len(case.expected_chunk_ids)
        )
        return RetrievalCaseMetrics(
            query=case.query,
            precision_at_k=relevant_results / k,
            recall_at_k=len(matched_targets) / expected_count,
            reciprocal_rank=reciprocal_rank,
        )
