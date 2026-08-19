"""Local Ollama implementation of the provider-neutral LLM contract."""

import math
from time import perf_counter
from typing import Any, Mapping
from urllib.parse import urlparse

import requests

from knowledge.llm import GenerationResult
from knowledge.llm_errors import (
    LLMInvocationError,
    LLMModelUnavailableError,
    LLMThrottledError,
    LLMValidationError,
    MalformedLLMResponseError,
)


class OllamaUnavailableError(LLMModelUnavailableError):
    """The configured loopback Ollama service or model is unavailable."""


class OllamaTimeoutError(OllamaUnavailableError):
    """The local Ollama request exceeded its configured timeout."""


class OllamaLLMProvider:
    """Generate text through a loopback-only Ollama HTTP endpoint."""

    provider_name = "ollama"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model_id: str = "gpt-oss:20b",
        connect_timeout_seconds: float = 5.0,
        read_timeout_seconds: float = 120.0,
        temperature: float = 0.0,
        maximum_tokens: int = 2_048,
        http_session: Any | None = None,
    ) -> None:
        self.base_url = self._validate_base_url(base_url)
        self.model_id = model_id.strip()
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        self.connect_timeout_seconds = self._positive_timeout(
            connect_timeout_seconds,
            "connect_timeout_seconds",
        )
        self.read_timeout_seconds = self._positive_timeout(
            read_timeout_seconds,
            "read_timeout_seconds",
        )
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0.0 <= float(temperature) <= 1.0
        ):
            raise ValueError("temperature must be between zero and one")
        if (
            isinstance(maximum_tokens, bool)
            or not isinstance(maximum_tokens, int)
            or maximum_tokens <= 0
        ):
            raise ValueError("maximum_tokens must be greater than zero")
        self.temperature = float(temperature)
        self.maximum_tokens = maximum_tokens
        self._session = http_session or requests.Session()

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_parameters: Mapping[str, Any] | None = None,
    ) -> GenerationResult:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt cannot be empty")
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("user_prompt cannot be empty")

        parameters = dict(model_parameters or {})
        temperature = parameters.pop("temperature", self.temperature)
        maximum_tokens = parameters.pop(
            "maximum_tokens",
            self.maximum_tokens,
        )
        response_format = parameters.pop("response_format", None)
        if parameters:
            raise LLMValidationError(
                "Unsupported Ollama model parameters were provided"
            )
        self._validate_generation_parameters(
            temperature,
            maximum_tokens,
            response_format,
        )

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": maximum_tokens,
            },
        }
        if response_format is not None:
            payload["format"] = response_format

        started = perf_counter()
        try:
            response = self._session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=(
                    self.connect_timeout_seconds,
                    self.read_timeout_seconds,
                ),
            )
        except requests.Timeout as error:
            raise OllamaTimeoutError(
                "The local Ollama request timed out"
            ) from error
        except requests.ConnectionError as error:
            raise OllamaUnavailableError(
                "The local Ollama service is unavailable"
            ) from error
        except requests.RequestException as error:
            raise LLMInvocationError(
                "The local Ollama request failed"
            ) from error

        self._raise_for_status(response)
        return self._parse_response(
            response,
            elapsed_ms=(perf_counter() - started) * 1_000,
        )

    def _parse_response(
        self,
        response: Any,
        *,
        elapsed_ms: float,
    ) -> GenerationResult:
        try:
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("response must be an object")
            message = payload["message"]
            if not isinstance(message, dict):
                raise ValueError("message must be an object")
            generated_text = message["content"]
            if (
                not isinstance(generated_text, str)
                or not generated_text.strip()
            ):
                raise ValueError("response contains no generated text")
            if payload.get("done") is not True:
                raise ValueError("response is not complete")
            response_model = payload.get("model", self.model_id)
            if not isinstance(response_model, str) or not response_model.strip():
                raise ValueError("model must be a string")
            input_tokens = self._optional_token_count(
                payload.get("prompt_eval_count")
            )
            output_tokens = self._optional_token_count(
                payload.get("eval_count")
            )
            finish_reason = payload.get("done_reason")
            if finish_reason is not None and not isinstance(
                finish_reason,
                str,
            ):
                raise ValueError("done_reason must be a string")
        except (
            KeyError,
            TypeError,
            ValueError,
            requests.JSONDecodeError,
        ) as error:
            raise MalformedLLMResponseError(
                "Ollama returned a malformed generation response"
            ) from error

        return GenerationResult(
            generated_text=generated_text.strip(),
            model_id=response_model.strip(),
            input_token_count=input_tokens,
            output_token_count=output_tokens,
            finish_reason=finish_reason,
            latency_ms=elapsed_ms,
            provider_metadata={"provider": self.provider_name},
        )

    @staticmethod
    def _raise_for_status(response: Any) -> None:
        status_code = getattr(response, "status_code", None)
        if status_code == 200:
            return
        if status_code == 429:
            raise LLMThrottledError(
                "The local Ollama service is busy; retry later"
            )
        if status_code == 404:
            raise OllamaUnavailableError(
                "The configured Ollama model is unavailable"
            )
        if status_code in {400, 422}:
            raise LLMValidationError(
                "Ollama rejected the generation request"
            )
        raise LLMInvocationError(
            "Ollama generation failed"
        )

    @staticmethod
    def _validate_generation_parameters(
        temperature: Any,
        maximum_tokens: Any,
        response_format: Any,
    ) -> None:
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0.0 <= float(temperature) <= 1.0
        ):
            raise LLMValidationError(
                "temperature must be between zero and one"
            )
        if (
            isinstance(maximum_tokens, bool)
            or not isinstance(maximum_tokens, int)
            or maximum_tokens <= 0
        ):
            raise LLMValidationError(
                "maximum_tokens must be a positive integer"
            )
        if response_format is not None and not (
            response_format == "json"
            or isinstance(response_format, dict)
        ):
            raise LLMValidationError(
                "response_format must be json or a JSON schema"
            )

    @staticmethod
    def _validate_base_url(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("base_url must be a string")
        normalized = value.strip().rstrip("/")
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("base_url must use http or https")
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("base_url must use a local loopback host")
        if (
            parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("base_url must not contain credentials or a path")
        if parsed.port is None:
            raise ValueError("base_url must include an explicit port")
        return normalized

    @staticmethod
    def _positive_timeout(value: Any, name: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"{name} must be positive")
        return float(value)

    @staticmethod
    def _optional_token_count(value: Any) -> int | None:
        if value is None:
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError("token count must be a non-negative integer")
        return value
