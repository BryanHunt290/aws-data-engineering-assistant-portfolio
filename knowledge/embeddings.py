"""Embedding abstraction without a concrete Bedrock integration."""

from enum import StrEnum
from typing import Protocol, Sequence, runtime_checkable


class EmbeddingStatus(StrEnum):
    """Manifest states used before and after embedding generation."""

    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Provider-neutral interface for generating embedding vectors."""

    @property
    def provider_name(self) -> str:
        """Return a stable provider identifier."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector for each input text."""


# Concrete Bedrock and deterministic local providers implement this unchanged
# interface in separate modules. Ingestion itself still invokes neither one.
