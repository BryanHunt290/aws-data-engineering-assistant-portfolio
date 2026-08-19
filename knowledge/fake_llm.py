"""Deterministic language-model provider for tests and local development."""

from typing import Any, Mapping

from knowledge.llm import GenerationResult
from knowledge.llm_errors import LLMInvocationError


class DeterministicFakeLLMProvider:
    """Return configured stable results without network access."""

    provider_name = "deterministic-fake"

    def __init__(
        self,
        *,
        response_text: str = "Deterministic test response.",
        model_id: str = "fake-llm-v1",
        responses_by_query: Mapping[str, str] | None = None,
        simulated_error: Exception | None = None,
        finish_reason: str = "end_turn",
        provider_metadata: Mapping[str, Any] | None = None,
        insufficient_context: bool = False,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id cannot be empty")
        self.model_id = model_id.strip()
        self.response_text = response_text
        self.responses_by_query = dict(responses_by_query or {})
        self.simulated_error = simulated_error
        self.finish_reason = (
            "insufficient_context"
            if insufficient_context
            else finish_reason
        )
        self.provider_metadata = dict(provider_metadata or {})
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_parameters: Mapping[str, Any] | None = None,
    ) -> GenerationResult:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "model_parameters": dict(model_parameters or {}),
            }
        )
        if self.simulated_error is not None:
            raise self.simulated_error
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise LLMInvocationError("System prompt cannot be empty")
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise LLMInvocationError("User prompt cannot be empty")

        selected = self.response_text
        for query_fragment in sorted(self.responses_by_query):
            if query_fragment in user_prompt:
                selected = self.responses_by_query[query_fragment]
                break
        return GenerationResult(
            generated_text=selected,
            model_id=self.model_id,
            input_token_count=len(
                (system_prompt + " " + user_prompt).split()
            ),
            output_token_count=len(selected.split()),
            finish_reason=self.finish_reason,
            latency_ms=0.0,
            provider_metadata=self.provider_metadata,
        )
