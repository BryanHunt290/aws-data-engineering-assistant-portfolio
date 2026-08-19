"""Deterministic scoped hybrid retrieval with reciprocal rank fusion."""

from dataclasses import dataclass
import math
from typing import Protocol, Sequence, runtime_checkable

from knowledge.keyword_retrieval import BM25Config, InMemoryBM25Retriever
from knowledge.retrieval import (
    InMemoryCosineRetriever,
    RetrievalEntry,
    RetrievalResult,
)


@runtime_checkable
class HybridRetriever(Protocol):
    """Provider-neutral interface for scoped text/vector fusion."""

    def retrieve(
        self,
        query_vector: Sequence[float],
        query_text: str,
        *,
        client_id: str,
        environment: str,
        top_k: int | None = None,
        minimum_score: float | None = None,
    ) -> list[RetrievalResult]:
        """Return ranked hybrid matches within one explicit scope."""


@dataclass(frozen=True)
class ReciprocalRankFusionConfig:
    """Validated reciprocal-rank fusion settings."""

    semantic_weight: float = 1.0
    keyword_weight: float = 1.0
    rank_constant: int = 60
    candidate_pool_size: int = 50
    top_k: int = 5
    minimum_score: float = 0.0

    def __post_init__(self) -> None:
        _validate_weight(self.semantic_weight, "semantic_weight")
        _validate_weight(self.keyword_weight, "keyword_weight")
        _validate_positive_integer(self.rank_constant, "rank_constant")
        _validate_positive_integer(
            self.candidate_pool_size,
            "candidate_pool_size",
        )
        _validate_positive_integer(self.top_k, "top_k")
        if self.candidate_pool_size < self.top_k:
            raise ValueError(
                "candidate_pool_size must be greater than or equal to top_k"
            )
        _validate_score(self.minimum_score)


class ReciprocalRankFusionRetriever:
    """Fuse local semantic and BM25 rankings within an explicit scope."""

    def __init__(
        self,
        entries: Sequence[RetrievalEntry] = (),
        *,
        config: ReciprocalRankFusionConfig | None = None,
        keyword_config: BM25Config | None = None,
    ) -> None:
        self.config = config or ReciprocalRankFusionConfig()
        self.keyword_config = keyword_config or BM25Config(
            top_k=self.config.candidate_pool_size,
        )
        self._entries: dict[tuple[str, str], RetrievalEntry] = {}
        for entry in entries:
            self.upsert(entry)

    def upsert(self, entry: RetrievalEntry) -> None:
        """Insert or replace one document/chunk pair."""

        key = (
            entry.embedding_record.document_id,
            entry.embedding_record.chunk_id,
        )
        self._entries[key] = entry

    def retrieve(
        self,
        query_vector: Sequence[float],
        query_text: str,
        *,
        client_id: str,
        environment: str,
        top_k: int | None = None,
        minimum_score: float | None = None,
    ) -> list[RetrievalResult]:
        """Return an RRF ranking without mixing client or environment data."""

        scope_client = _validate_scope(client_id, "client_id")
        scope_environment = _validate_scope(environment, "environment")
        limit = self.config.top_k if top_k is None else top_k
        threshold = (
            self.config.minimum_score
            if minimum_score is None
            else minimum_score
        )
        _validate_positive_integer(limit, "top_k")
        _validate_score(threshold)

        scoped_entries = [
            entry
            for entry in self._entries.values()
            if (
                entry.metadata.get("client_id") == scope_client
                and entry.metadata.get("environment") == scope_environment
            )
        ]
        if not scoped_entries:
            return []

        candidate_limit = min(
            self.config.candidate_pool_size,
            len(scoped_entries),
        )
        semantic_results = InMemoryCosineRetriever(
            scoped_entries,
            top_k=candidate_limit,
            minimum_similarity=-1.0,
        ).retrieve(
            query_vector,
            top_k=candidate_limit,
            minimum_similarity=-1.0,
        )
        keyword_results = InMemoryBM25Retriever(
            scoped_entries,
            config=self.keyword_config,
        ).retrieve(
            query_text,
            client_id=scope_client,
            environment=scope_environment,
            top_k=candidate_limit,
            minimum_score=0.0,
        )

        result_by_key: dict[tuple[str, str], RetrievalResult] = {}
        fused_scores: dict[tuple[str, str], float] = {}
        component_ranks: dict[
            tuple[str, str],
            dict[str, int | float],
        ] = {}

        for strategy, weight, results in (
            (
                "semantic",
                float(self.config.semantic_weight),
                semantic_results,
            ),
            (
                "keyword",
                float(self.config.keyword_weight),
                keyword_results,
            ),
        ):
            for rank, result in enumerate(results, start=1):
                key = (result.document_id, result.chunk_id)
                result_by_key.setdefault(key, result)
                fused_scores[key] = fused_scores.get(key, 0.0) + (
                    weight / (self.config.rank_constant + rank)
                )
                component_ranks.setdefault(key, {})[
                    f"{strategy}_rank"
                ] = rank
                component_ranks[key][f"{strategy}_score"] = (
                    result.similarity_score
                )

        ranked: list[RetrievalResult] = []
        for key, score in fused_scores.items():
            if score < float(threshold):
                continue
            source_result = result_by_key[key]
            metadata = dict(source_result.metadata)
            metadata["fusion_method"] = "reciprocal_rank_fusion"
            metadata.update(component_ranks[key])
            ranked.append(
                RetrievalResult(
                    document_id=source_result.document_id,
                    chunk_id=source_result.chunk_id,
                    source=source_result.source,
                    text=source_result.text,
                    similarity_score=score,
                    metadata=metadata,
                )
            )

        ranked.sort(
            key=lambda result: (
                -result.similarity_score,
                result.document_id,
                result.chunk_id,
            )
        )
        return ranked[:limit]


def _validate_weight(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0.0
    ):
        raise ValueError(f"{name} must be a finite number greater than zero")


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _validate_score(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0.0
    ):
        raise ValueError(
            "minimum_score must be a finite number greater than or equal to zero"
        )


def _validate_scope(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")
    return value.strip()
