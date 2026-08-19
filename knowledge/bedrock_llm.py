"""Amazon Bedrock Converse implementation of the LLM provider contract."""

import math
from time import perf_counter
from typing import Any, Mapping

from knowledge.llm import GenerationResult
from knowledge.llm_errors import (
    LLMAccessDeniedError,
    LLMInvocationError,
    LLMModelUnavailableError,
    LLMThrottledError,
    LLMValidationError,
    MalformedLLMResponseError,
)


class BedrockLLMProvider:
    """Generate responses through Bedrock Runtime's Converse API."""

    provider_name = "amazon-bedrock"

    def __init__(
        self,
        *,
        model_id: str,
        region_name: str,
        temperature: float = 0.1,
        maximum_tokens: int = 1_024,
        timeout_seconds: float = 30.0,
        bedrock_runtime_client: Any | None = None,
    ) -> None:
        self.model_id = model_id.strip()
        self.region_name = region_name.strip().lower()
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if not self.region_name:
            raise ValueError("region_name cannot be empty")
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("temperature must be between zero and one")
        if (
            isinstance(maximum_tokens, bool)
            or not isinstance(maximum_tokens, int)
            or maximum_tokens <= 0
        ):
            raise ValueError("maximum_tokens must be greater than zero")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")

        self.temperature = temperature
        self.maximum_tokens = maximum_tokens
        self.timeout_seconds = float(timeout_seconds)
        self._client = bedrock_runtime_client or self._create_client()

    def _create_client(self) -> Any:
        import boto3
        from botocore.config import Config

        return boto3.client(
            "bedrock-runtime",
            region_name=self.region_name,
            config=Config(
                connect_timeout=self.timeout_seconds,
                read_timeout=self.timeout_seconds,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

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
        if parameters:
            raise LLMValidationError(
                "Unsupported model parameters were provided"
            )
        if not isinstance(temperature, (int, float)) or isinstance(
            temperature,
            bool,
        ) or not 0.0 <= float(temperature) <= 1.0:
            raise LLMValidationError(
                "temperature must be between zero and one"
            )
        if (
            not isinstance(maximum_tokens, int)
            or isinstance(maximum_tokens, bool)
            or maximum_tokens <= 0
        ):
            raise LLMValidationError(
                "maximum_tokens must be a positive integer"
            )

        started = perf_counter()
        try:
            response = self._client.converse(
                modelId=self.model_id,
                system=[{"text": system_prompt}],
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": user_prompt}],
                    }
                ],
                inferenceConfig={
                    "temperature": float(temperature),
                    "maxTokens": maximum_tokens,
                },
            )
        except Exception as error:
            raise self._translate_error(error) from error

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
            if not isinstance(response, dict):
                raise ValueError("response must be an object")
            output = response["output"]
            message = output["message"]
            content = message["content"]
            if not isinstance(content, list) or not content:
                raise ValueError("content must be non-empty")
            text_parts = [
                item["text"]
                for item in content
                if isinstance(item, dict)
                and isinstance(item.get("text"), str)
            ]
            generated_text = "\n".join(text_parts).strip()
            if not generated_text:
                raise ValueError("response contains no generated text")
            usage = response.get("usage", {})
            if not isinstance(usage, dict):
                raise ValueError("usage must be an object")
            input_tokens = self._optional_token_count(
                usage.get("inputTokens")
            )
            output_tokens = self._optional_token_count(
                usage.get("outputTokens")
            )
            cache_read_tokens = self._optional_token_count(
                usage.get("cacheReadInputTokens")
            )
            cache_write_tokens = self._optional_token_count(
                usage.get("cacheWriteInputTokens")
            )
            finish_reason = response.get("stopReason")
            if finish_reason is not None and not isinstance(
                finish_reason,
                str,
            ):
                raise ValueError("stopReason must be a string")
            response_metadata = response.get("ResponseMetadata", {})
            request_id = (
                response_metadata.get("RequestId")
                if isinstance(response_metadata, dict)
                else None
            )
            provider_metadata = {
                "provider": self.provider_name,
                "request_id": request_id,
            }
            if cache_read_tokens is not None:
                provider_metadata["cache_read_token_count"] = (
                    cache_read_tokens
                )
            if cache_write_tokens is not None:
                provider_metadata["cache_write_token_count"] = (
                    cache_write_tokens
                )
            return GenerationResult(
                generated_text=generated_text,
                model_id=self.model_id,
                input_token_count=input_tokens,
                output_token_count=output_tokens,
                finish_reason=finish_reason,
                latency_ms=elapsed_ms,
                provider_metadata=provider_metadata,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise MalformedLLMResponseError(
                "Bedrock returned a malformed generation response"
            ) from error

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

    def _translate_error(self, error: Exception) -> Exception:
        code = self._error_code(error)
        if code in {"ThrottlingException", "TooManyRequestsException"}:
            return LLMThrottledError(
                "Bedrock generation request was throttled"
            )
        if code in {
            "AccessDeniedException",
            "UnauthorizedException",
            "UnrecognizedClientException",
        }:
            return LLMAccessDeniedError(
                "Access to the Bedrock language model was denied"
            )
        if code in {
            "ModelNotReadyException",
            "ModelTimeoutException",
            "ResourceNotFoundException",
            "ServiceUnavailableException",
        }:
            return LLMModelUnavailableError(
                "The Bedrock language model is unavailable"
            )
        if code == "ValidationException":
            return LLMValidationError(
                "Bedrock rejected the generation request"
            )
        return LLMInvocationError(
            f"Bedrock generation failed ({code or 'unknown'})"
        )

    @staticmethod
    def _error_code(error: Exception) -> str | None:
        response = getattr(error, "response", None)
        if not isinstance(response, dict):
            return None
        details = response.get("Error")
        if not isinstance(details, dict):
            return None
        code = details.get("Code")
        return str(code) if code else None
