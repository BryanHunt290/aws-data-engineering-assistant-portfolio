"""Lazy, cached vector-store credential resolution."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from typing import Any, Protocol, runtime_checkable

from knowledge.indexing_errors import (
    IndexingSecretError,
    IndexingSecretSchemaError,
)


@dataclass(frozen=True)
class QdrantCredentials:
    """Validated Qdrant authentication material held only in memory."""

    api_key: str
    endpoint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise IndexingSecretSchemaError(
                "Vector-store secret must contain a non-empty api_key"
            )
        endpoint = self.endpoint
        if endpoint is not None and (
            not isinstance(endpoint, str) or not endpoint.strip()
        ):
            raise IndexingSecretSchemaError(
                "Vector-store secret endpoint must be a non-empty string"
            )
        object.__setattr__(self, "api_key", self.api_key.strip())
        object.__setattr__(
            self,
            "endpoint",
            endpoint.strip() if endpoint is not None else None,
        )


@runtime_checkable
class QdrantCredentialResolver(Protocol):
    """Provider-neutral boundary for resolving Qdrant credentials."""

    def resolve(self) -> QdrantCredentials:
        """Resolve and cache validated credentials."""


class SecretsManagerQdrantCredentialResolver:
    """Resolve one JSON secret lazily through an injected AWS client."""

    def __init__(
        self,
        secret_identifier: str,
        *,
        secrets_client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        if not isinstance(secret_identifier, str) or not secret_identifier.strip():
            raise ValueError("secret_identifier cannot be empty")
        if secrets_client is not None and client_factory is not None:
            raise ValueError("Provide secrets_client or client_factory, not both")
        self._secret_identifier = secret_identifier.strip()
        self._client = secrets_client
        self._client_factory = client_factory or self._default_client
        self._cached: QdrantCredentials | None = None

    def resolve(self) -> QdrantCredentials:
        if self._cached is not None:
            return self._cached
        try:
            response = self._get_client().get_secret_value(
                SecretId=self._secret_identifier
            )
        except Exception:
            raise IndexingSecretError(
                "Vector-store credentials could not be retrieved"
            ) from None
        credentials = self._parse_response(response)
        self._cached = credentials
        return credentials

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    @staticmethod
    def _default_client() -> Any:
        import boto3

        return boto3.client("secretsmanager")

    @staticmethod
    def _parse_response(response: object) -> QdrantCredentials:
        if not isinstance(response, Mapping):
            raise IndexingSecretSchemaError(
                "Vector-store secret response is malformed"
            )
        raw_secret = response.get("SecretString")
        if not isinstance(raw_secret, str) or not raw_secret:
            raise IndexingSecretSchemaError(
                "Vector-store secret must use SecretString JSON"
            )
        try:
            payload = json.loads(raw_secret)
        except json.JSONDecodeError as error:
            raise IndexingSecretSchemaError(
                "Vector-store secret must contain valid JSON"
            ) from error
        if not isinstance(payload, Mapping):
            raise IndexingSecretSchemaError(
                "Vector-store secret must contain a JSON object"
            )
        allowed = {"api_key", "endpoint"}
        if set(payload) - allowed:
            raise IndexingSecretSchemaError(
                "Vector-store secret contains unsupported fields"
            )
        api_key = payload.get("api_key")
        endpoint = payload.get("endpoint")
        if not isinstance(api_key, str):
            raise IndexingSecretSchemaError(
                "Vector-store secret must contain a string api_key"
            )
        if endpoint is not None and not isinstance(endpoint, str):
            raise IndexingSecretSchemaError(
                "Vector-store secret endpoint must be a string"
            )
        return QdrantCredentials(api_key=api_key, endpoint=endpoint)
