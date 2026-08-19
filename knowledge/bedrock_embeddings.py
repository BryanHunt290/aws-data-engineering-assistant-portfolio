"""Amazon Bedrock implementation of the existing embedding interface."""

import json
import math
from typing import Any, Sequence

from knowledge.embedding_errors import (
    EmbeddingAccessDeniedError,
    EmbeddingInvocationError,
    EmbeddingModelUnavailableError,
    EmbeddingThrottledError,
    MalformedEmbeddingResponseError,
)


class BedrockEmbeddingProvider:
    """Invoke an Amazon Titan text embedding model through Bedrock Runtime."""

    provider_name = "amazon-bedrock"

    def __init__(
        self,
        *,
        model_id: str,
        region_name: str,
        bedrock_runtime_client: Any | None = None,
        dimensions: int | None = None,
        normalize: bool = True,
    ) -> None:
        self.model_id = model_id.strip()
        self.region_name = region_name.strip()
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if not self.region_name:
            raise ValueError("region_name cannot be empty")
        if dimensions is not None and dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")

        self._dimensions = dimensions
        self._normalize = normalize
        self._client = bedrock_runtime_client

    def _create_client(self) -> Any:
        import boto3

        return boto3.client(
            "bedrock-runtime",
            region_name=self.region_name,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts without logging inputs or returned vectors."""

        vectors: list[list[float]] = []
        for text in texts:
            if not isinstance(text, str) or not text:
                raise ValueError("Embedding input text cannot be empty")
            vectors.append(self._invoke_one(text))
        return vectors

    def _invoke_one(self, text: str) -> list[float]:
        request: dict[str, object] = {
            "inputText": text,
            "normalize": self._normalize,
        }
        if self._dimensions is not None:
            request["dimensions"] = self._dimensions

        try:
            response = self._get_client().invoke_model(
                modelId=self.model_id,
                body=json.dumps(request).encode("utf-8"),
                contentType="application/json",
                accept="application/json",
            )
        except Exception as error:
            raise self._translate_error(error) from error

        try:
            body = response["body"]
            raw_body = body.read() if hasattr(body, "read") else body
            if isinstance(raw_body, bytes):
                raw_body = raw_body.decode("utf-8")
            payload = json.loads(raw_body)
            vector = payload["embedding"]
            if not isinstance(vector, list) or not vector:
                raise ValueError("embedding must be a non-empty list")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                for value in vector
            ):
                raise ValueError("embedding must contain numbers")
            parsed = [float(value) for value in vector]
            if not all(math.isfinite(value) for value in parsed):
                raise ValueError("embedding contains a non-finite value")
            if self._dimensions is not None and (
                len(parsed) != self._dimensions
            ):
                raise ValueError("embedding dimensions do not match request")
            return parsed
        except (
            KeyError,
            TypeError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:
            raise MalformedEmbeddingResponseError(
                "Bedrock returned a malformed embedding response"
            ) from error

    def _translate_error(self, error: Exception) -> Exception:
        code = self._error_code(error)
        if code in {"ThrottlingException", "TooManyRequestsException"}:
            return EmbeddingThrottledError(
                "Bedrock embedding request was throttled"
            )
        if code in {
            "AccessDeniedException",
            "UnauthorizedException",
            "UnrecognizedClientException",
        }:
            return EmbeddingAccessDeniedError(
                "Access to the Bedrock embedding model was denied"
            )
        if code in {
            "ModelNotReadyException",
            "ModelTimeoutException",
            "ResourceNotFoundException",
            "ServiceUnavailableException",
        }:
            return EmbeddingModelUnavailableError(
                "The Bedrock embedding model is unavailable"
            )
        return EmbeddingInvocationError(
            f"Bedrock embedding invocation failed ({code or 'unknown'})"
        )

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._create_client()
        return self._client

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
