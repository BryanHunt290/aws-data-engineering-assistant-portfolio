"""Provider-neutral retrieval and local cosine-similarity implementation."""

from dataclasses import dataclass
import math
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from knowledge.models import EmbeddingRecord


@dataclass(frozen=True)
class RetrievalEntry:
    """Embedding plus safe display fields used by a retriever."""

    embedding_record: EmbeddingRecord
    source: str
    text: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RetrievalResult:
    """Ranked retrieval result returned to assistant orchestration."""

    document_id: str
    chunk_id: str
    source: str
    text: str
    similarity_score: float
    metadata: Mapping[str, Any]


@runtime_checkable
class Retriever(Protocol):
    """Provider-neutral vector retrieval interface."""

    def retrieve(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int | None = None,
        minimum_similarity: float | None = None,
    ) -> list[RetrievalResult]:
        """Return ranked results for a query vector."""


class InMemoryCosineRetriever:
    """Deterministic local retriever for development and evaluation."""

    def __init__(
        self,
        entries: Sequence[RetrievalEntry] = (),
        *,
        top_k: int = 5,
        minimum_similarity: float = 0.0,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not -1.0 <= minimum_similarity <= 1.0:
            raise ValueError(
                "minimum_similarity must be between -1 and 1"
            )
        self._top_k = top_k
        self._minimum_similarity = minimum_similarity
        self._entries: dict[tuple[str, str], RetrievalEntry] = {}
        for entry in entries:
            self.upsert(entry)

    def upsert(self, entry: RetrievalEntry) -> None:
        self._validate_vector(entry.embedding_record.embedding_vector)
        key = (
            entry.embedding_record.document_id,
            entry.embedding_record.chunk_id,
        )
        self._entries[key] = entry

    def retrieve(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int | None = None,
        minimum_similarity: float | None = None,
    ) -> list[RetrievalResult]:
        limit = self._top_k if top_k is None else top_k
        threshold = (
            self._minimum_similarity
            if minimum_similarity is None
            else minimum_similarity
        )
        if limit <= 0:
            raise ValueError("top_k must be greater than zero")
        if not -1.0 <= threshold <= 1.0:
            raise ValueError(
                "minimum_similarity must be between -1 and 1"
            )
        query = self._validate_vector(query_vector)

        ranked: list[RetrievalResult] = []
        for entry in self._entries.values():
            vector = entry.embedding_record.embedding_vector
            if len(vector) != len(query):
                raise ValueError(
                    "Query and stored embedding dimensions must match"
                )
            score = self._cosine_similarity(query, vector)
            if score < threshold:
                continue
            ranked.append(
                RetrievalResult(
                    document_id=entry.embedding_record.document_id,
                    chunk_id=entry.embedding_record.chunk_id,
                    source=entry.source,
                    text=entry.text,
                    similarity_score=score,
                    metadata=dict(entry.metadata),
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

    @staticmethod
    def _validate_vector(vector: Sequence[float]) -> tuple[float, ...]:
        if not vector:
            raise ValueError("Vector cannot be empty")
        parsed = tuple(float(value) for value in vector)
        if not all(math.isfinite(value) for value in parsed):
            raise ValueError("Vector contains a non-finite value")
        if math.isclose(
            math.sqrt(sum(value * value for value in parsed)),
            0.0,
        ):
            raise ValueError("Vector magnitude cannot be zero")
        return parsed

    @staticmethod
    def _cosine_similarity(
        left: Sequence[float],
        right: Sequence[float],
    ) -> float:
        dot_product = sum(
            left_value * right_value
            for left_value, right_value in zip(left, right, strict=True)
        )
        left_magnitude = math.sqrt(
            sum(value * value for value in left)
        )
        right_magnitude = math.sqrt(
            sum(value * value for value in right)
        )
        return dot_product / (left_magnitude * right_magnitude)
