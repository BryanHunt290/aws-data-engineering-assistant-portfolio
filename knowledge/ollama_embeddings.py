"""Local Ollama implementation of the embedding provider contract."""

import math
from typing import Any, Sequence
from urllib.parse import urlparse

import requests

from knowledge.embedding_errors import (
    EmbeddingInvocationError,
    EmbeddingThrottledError,
    MalformedEmbeddingResponseError,
    OllamaEmbeddingTimeoutError,
    OllamaEmbeddingUnavailableError,
)


class OllamaEmbeddingProvider:
    """Generate bounded embedding batches through loopback Ollama HTTP."""

    provider_name = "ollama"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model_id: str = "embeddinggemma",
        connect_timeout_seconds: float = 5.0,
        read_timeout_seconds: float = 120.0,
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
        self._session = http_session or requests.Session()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one validated vector per non-empty input string."""

        if not texts:
            raise ValueError("texts cannot be empty")
        inputs = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in inputs):
            raise ValueError("Embedding text cannot be empty")

        try:
            response = self._session.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model_id,
                    "input": [text.strip() for text in inputs],
                },
                timeout=(
                    self.connect_timeout_seconds,
                    self.read_timeout_seconds,
                ),
            )
        except requests.Timeout as error:
            raise OllamaEmbeddingTimeoutError(
                "The local Ollama embedding request timed out"
            ) from error
        except requests.ConnectionError as error:
            raise OllamaEmbeddingUnavailableError(
                "The local Ollama service is unavailable"
            ) from error
        except requests.RequestException as error:
            raise EmbeddingInvocationError(
                "The local Ollama embedding request failed"
            ) from error

        self._raise_for_status(response)
        return self._parse_response(response, expected_count=len(inputs))

    def embed_text(self, text: str) -> list[float]:
        """Convenience wrapper for callers embedding exactly one string."""

        return self.embed([text])[0]

    def check_connection(self) -> None:
        """Perform a bounded, explicit local service connectivity check."""

        try:
            response = self._session.get(
                f"{self.base_url}/api/version",
                timeout=(
                    self.connect_timeout_seconds,
                    self.read_timeout_seconds,
                ),
            )
        except requests.Timeout as error:
            raise OllamaEmbeddingTimeoutError(
                "The local Ollama connection check timed out"
            ) from error
        except requests.RequestException as error:
            raise OllamaEmbeddingUnavailableError(
                "The local Ollama service is unavailable"
            ) from error
        if getattr(response, "status_code", None) != 200:
            raise OllamaEmbeddingUnavailableError(
                "The local Ollama service is unavailable"
            )
        try:
            payload = response.json()
            if (
                not isinstance(payload, dict)
                or not isinstance(payload.get("version"), str)
                or not payload["version"].strip()
            ):
                raise ValueError("version is missing")
        except (
            TypeError,
            ValueError,
            requests.JSONDecodeError,
        ) as error:
            raise MalformedEmbeddingResponseError(
                "Ollama returned a malformed connection response"
            ) from error

    def _parse_response(
        self,
        response: Any,
        *,
        expected_count: int,
    ) -> list[list[float]]:
        try:
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("response must be an object")
            response_model = payload.get("model")
            if (
                not isinstance(response_model, str)
                or not response_model.strip()
            ):
                raise ValueError("model must be a string")
            embeddings = payload["embeddings"]
            if (
                not isinstance(embeddings, list)
                or len(embeddings) != expected_count
            ):
                raise ValueError("embedding count does not match input")
            parsed = [self._parse_vector(vector) for vector in embeddings]
            dimensions = {len(vector) for vector in parsed}
            if len(dimensions) != 1:
                raise ValueError("embedding dimensions are inconsistent")
        except (
            KeyError,
            TypeError,
            ValueError,
            requests.JSONDecodeError,
        ) as error:
            raise MalformedEmbeddingResponseError(
                "Ollama returned a malformed embedding response"
            ) from error
        return parsed

    @staticmethod
    def _parse_vector(value: Any) -> list[float]:
        if not isinstance(value, list) or not value:
            raise ValueError("embedding must be a non-empty array")
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in value
        ):
            raise ValueError("embedding must contain only numbers")
        vector = [float(item) for item in value]
        if not all(math.isfinite(item) for item in vector):
            raise ValueError("embedding contains a non-finite number")
        return vector

    @staticmethod
    def _raise_for_status(response: Any) -> None:
        status_code = getattr(response, "status_code", None)
        if status_code == 200:
            return
        if status_code == 429:
            raise EmbeddingThrottledError(
                "The local Ollama service is busy; retry later"
            )
        if status_code == 404:
            raise OllamaEmbeddingUnavailableError(
                "The configured Ollama embedding model is unavailable"
            )
        raise EmbeddingInvocationError("Ollama embedding generation failed")

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
