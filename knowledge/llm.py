"""Provider-neutral language-model generation contracts."""

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class GenerationResult:
    """Typed output returned by an LLM provider."""

    generated_text: str
    model_id: str
    input_token_count: int | None = None
    output_token_count: int | None = None
    finish_reason: str | None = None
    latency_ms: float = 0.0
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.generated_text, str):
            raise ValueError("generated_text must be a string")
        if not self.model_id.strip():
            raise ValueError("model_id cannot be empty")
        for name, value in (
            ("input_token_count", self.input_token_count),
            ("output_token_count", self.output_token_count),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("latency_ms must be a finite non-negative value")
        object.__setattr__(self, "model_id", self.model_id.strip())
        object.__setattr__(
            self,
            "provider_metadata",
            dict(self.provider_metadata),
        )

    @property
    def indicates_insufficient_context(self) -> bool:
        """Return an explicit provider signal, never a semantic guess."""

        normalized_text = self.generated_text.strip().casefold()
        return (
            self.finish_reason == "insufficient_context"
            or self.provider_metadata.get("insufficient_context") is True
            or normalized_text.startswith("insufficient_context:")
            or normalized_text.startswith("[insufficient_context]")
        )


@runtime_checkable
class LLMProvider(Protocol):
    """Provider-neutral interface for text generation."""

    @property
    def provider_name(self) -> str:
        """Return a stable provider identifier."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_parameters: Mapping[str, Any] | None = None,
    ) -> GenerationResult:
        """Generate text without executing infrastructure actions."""
