"""Deterministic embedding provider for local development and tests."""

import hashlib
import math
from typing import Sequence

from knowledge.embedding_errors import EmbeddingInvocationError


class DeterministicFakeEmbeddingProvider:
    """Create stable normalized vectors without network calls."""

    provider_name = "deterministic-fake"

    def __init__(
        self,
        *,
        model_id: str = "fake-embedding-v1",
        dimensions: int = 16,
        fail_on_texts: frozenset[str] | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id cannot be empty")
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")
        self.model_id = model_id
        self.dimensions = dimensions
        self._fail_on_texts = fail_on_texts or frozenset()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            if text in self._fail_on_texts:
                raise EmbeddingInvocationError(
                    "Configured fake embedding failure"
                )
            vectors.append(self._vector(text))
        return vectors

    def _vector(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text:
            raise ValueError("Embedding input text cannot be empty")

        values: list[float] = []
        counter = 0
        while len(values) < self.dimensions:
            digest = hashlib.sha256(
                f"{self.model_id}:{counter}:{text}".encode("utf-8")
            ).digest()
            values.extend((byte / 127.5) - 1.0 for byte in digest)
            counter += 1
        vector = values[: self.dimensions]
        magnitude = math.sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector]
