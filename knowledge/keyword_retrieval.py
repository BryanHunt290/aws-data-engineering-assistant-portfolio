"""Provider-neutral, deterministic in-memory BM25 retrieval."""

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Protocol, Sequence, runtime_checkable

from knowledge.retrieval import RetrievalEntry, RetrievalResult


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@runtime_checkable
class KeywordRetriever(Protocol):
    """Provider-neutral interface for scoped keyword retrieval."""

    def retrieve(
        self,
        query_text: str,
        *,
        client_id: str,
        environment: str,
        top_k: int | None = None,
        minimum_score: float | None = None,
    ) -> list[RetrievalResult]:
        """Return ranked lexical matches within one explicit scope."""


@dataclass(frozen=True)
class BM25Config:
    """Validated BM25 and result-filtering settings."""

    k1: float = 1.5
    b: float = 0.75
    top_k: int = 5
    minimum_score: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.k1, bool)
            or not isinstance(self.k1, (int, float))
            or not math.isfinite(float(self.k1))
            or self.k1 <= 0
        ):
            raise ValueError("k1 must be a finite number greater than zero")
        if (
            isinstance(self.b, bool)
            or not isinstance(self.b, (int, float))
            or not math.isfinite(float(self.b))
            or not 0.0 <= float(self.b) <= 1.0
        ):
            raise ValueError("b must be a finite number between zero and one")
        _validate_limit(self.top_k)
        _validate_minimum_score(self.minimum_score)


class InMemoryBM25Retriever:
    """Rank local text with BM25 without dependencies or network access."""

    def __init__(
        self,
        entries: Sequence[RetrievalEntry] = (),
        *,
        config: BM25Config | None = None,
    ) -> None:
        self.config = config or BM25Config()
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
        query_text: str,
        *,
        client_id: str,
        environment: str,
        top_k: int | None = None,
        minimum_score: float | None = None,
    ) -> list[RetrievalResult]:
        """Return deterministic BM25 matches within the requested scope."""

        query_tokens = _tokenize(query_text)
        if not query_tokens:
            raise ValueError("query_text must contain at least one token")
        scope_client = _validate_scope_value(client_id, "client_id")
        scope_environment = _validate_scope_value(
            environment,
            "environment",
        )
        limit = self.config.top_k if top_k is None else top_k
        threshold = (
            self.config.minimum_score
            if minimum_score is None
            else minimum_score
        )
        _validate_limit(limit)
        _validate_minimum_score(threshold)

        scoped_entries = [
            entry
            for entry in self._entries.values()
            if _entry_matches_scope(
                entry,
                client_id=scope_client,
                environment=scope_environment,
            )
        ]
        if not scoped_entries:
            return []

        token_counts = [
            Counter(_tokenize(entry.text))
            for entry in scoped_entries
        ]
        document_lengths = [
            sum(counts.values())
            for counts in token_counts
        ]
        average_length = sum(document_lengths) / len(document_lengths)
        if math.isclose(average_length, 0.0):
            return []
        query_counts = Counter(query_tokens)
        document_frequency = {
            token: sum(
                1 for counts in token_counts if counts.get(token, 0) > 0
            )
            for token in query_counts
        }

        ranked: list[RetrievalResult] = []
        for entry, counts, document_length in zip(
            scoped_entries,
            token_counts,
            document_lengths,
            strict=True,
        ):
            score = self._score(
                query_counts=query_counts,
                document_counts=counts,
                document_frequency=document_frequency,
                document_count=len(scoped_entries),
                document_length=document_length,
                average_length=average_length,
            )
            # A zero BM25 score means there was no lexical match. Returning
            # such entries would make no-result behavior misleading.
            if score <= 0.0 or score < float(threshold):
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

    def _score(
        self,
        *,
        query_counts: Counter[str],
        document_counts: Counter[str],
        document_frequency: dict[str, int],
        document_count: int,
        document_length: int,
        average_length: float,
    ) -> float:
        score = 0.0
        for token, query_frequency in query_counts.items():
            term_frequency = document_counts.get(token, 0)
            if term_frequency == 0:
                continue
            frequency = document_frequency[token]
            inverse_document_frequency = math.log(
                1.0
                + (
                    document_count - frequency + 0.5
                )
                / (frequency + 0.5)
            )
            length_normalization = (
                1.0
                - float(self.config.b)
                + float(self.config.b)
                * document_length
                / average_length
            )
            numerator = term_frequency * (float(self.config.k1) + 1.0)
            denominator = (
                term_frequency
                + float(self.config.k1) * length_normalization
            )
            score += (
                query_frequency
                * inverse_document_frequency
                * numerator
                / denominator
            )
        return score


def _tokenize(text: str) -> tuple[str, ...]:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return tuple(_TOKEN_PATTERN.findall(text.casefold()))


def _entry_matches_scope(
    entry: RetrievalEntry,
    *,
    client_id: str,
    environment: str,
) -> bool:
    return (
        entry.metadata.get("client_id") == client_id
        and entry.metadata.get("environment") == environment
    )


def _validate_scope_value(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")
    return value.strip()


def _validate_limit(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("top_k must be greater than zero")


def _validate_minimum_score(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0.0
    ):
        raise ValueError(
            "minimum_score must be a finite number greater than or equal to zero"
        )
