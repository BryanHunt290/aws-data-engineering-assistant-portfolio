"""Validated composition for local and production automatic indexing."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Any
from urllib.parse import urlparse

from knowledge.bedrock_embeddings import BedrockEmbeddingProvider
from knowledge.embedding_workflow import EmbeddingWorkflow
from knowledge.fake_embeddings import DeterministicFakeEmbeddingProvider
from knowledge.indexing_errors import (
    IndexingConfigurationError,
    IndexingDependencyError,
)
from knowledge.indexing_secrets import (
    QdrantCredentialResolver,
    SecretsManagerQdrantCredentialResolver,
)
from knowledge.manifest import KnowledgeManifestRepository
from knowledge.media_classification import require_indexable_metadata
from knowledge.ollama_embeddings import OllamaEmbeddingProvider
from knowledge.qdrant_vector_store import QdrantVectorStore
from knowledge.retrieval import RetrievalEntry, RetrievalResult
from knowledge.storage import KnowledgeStorage
from knowledge.vector_indexing import VectorIndexingWorkflow
from knowledge.vector_store import InMemoryVectorStore, VectorStore


class IndexingEmbeddingProviderName(StrEnum):
    """Embedding implementations available to automatic indexing."""

    FAKE = "fake"
    OLLAMA = "ollama"
    BEDROCK = "bedrock"


class IndexingVectorStoreName(StrEnum):
    """Vector-store implementations available to automatic indexing."""

    MEMORY = "memory"
    QDRANT = "qdrant"


class IndexingRuntimeMode(StrEnum):
    """Security posture applied to indexing configuration."""

    LOCAL = "local"
    PRODUCTION = "production"


class VectorEndpointSource(StrEnum):
    """Approved sources for a non-credential Qdrant endpoint."""

    ENVIRONMENT = "environment"
    SECRET = "secret"


@dataclass(frozen=True)
class AutomaticIndexingConfig:
    """Environment-independent, fail-closed automatic indexing settings."""

    enabled: bool = False
    runtime_mode: IndexingRuntimeMode = IndexingRuntimeMode.LOCAL
    embedding_provider: IndexingEmbeddingProviderName | None = None
    vector_store: IndexingVectorStoreName | None = None
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_batch_size: int = 8
    embedding_dimensions: int | None = None
    aws_region: str = "us-west-2"
    ollama_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "embeddinggemma"
    qdrant_url: str | None = "http://localhost:6333"
    qdrant_collection: str = "dea_knowledge_embeddinggemma_v1"
    qdrant_api_key: str | None = None
    qdrant_secret_identifier: str | None = None
    endpoint_source: VectorEndpointSource = VectorEndpointSource.ENVIRONMENT
    tls_required: bool = False
    authentication_required: bool = False
    connect_timeout_seconds: float = 5.0
    request_timeout_seconds: float = 10.0
    retry_limit: int = 2
    manifest_conflict_retry_limit: int = 3
    maximum_descriptor_batch_size: int = 10
    maximum_chunks_per_invocation: int = 500
    client_id: str | None = None
    environment: str | None = None
    knowledge_namespace: str = "default"
    knowledge_domain: str = "general"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise IndexingConfigurationError("enabled must be a boolean")
        try:
            mode = IndexingRuntimeMode(self.runtime_mode)
            endpoint_source = VectorEndpointSource(self.endpoint_source)
            provider = (
                IndexingEmbeddingProviderName(self.embedding_provider)
                if self.embedding_provider is not None
                else None
            )
            store = (
                IndexingVectorStoreName(self.vector_store)
                if self.vector_store is not None
                else None
            )
        except ValueError as error:
            raise IndexingConfigurationError(
                "Unsupported automatic indexing configuration"
            ) from error
        self._validate_numbers()
        text = self._normalized_text()
        if self.enabled and (provider is None or store is None):
            raise IndexingConfigurationError(
                "Enabled automatic indexing requires explicit embedding and "
                "vector-store providers"
            )
        if self.enabled and mode == IndexingRuntimeMode.PRODUCTION:
            self._validate_scope(text)
        if store == IndexingVectorStoreName.QDRANT:
            self._validate_qdrant(mode, endpoint_source, text)
        if mode == IndexingRuntimeMode.PRODUCTION and self.enabled:
            if provider != IndexingEmbeddingProviderName.BEDROCK:
                raise IndexingConfigurationError(
                    "Production indexing requires the Bedrock embedding provider"
                )
            if store != IndexingVectorStoreName.QDRANT:
                raise IndexingConfigurationError(
                    "Production indexing requires the durable Qdrant vector store"
                )
            if not self.tls_required or not self.authentication_required:
                raise IndexingConfigurationError(
                    "Production Qdrant requires TLS and authentication"
                )
            if self.qdrant_api_key:
                raise IndexingConfigurationError(
                    "Production credentials must use a secret reference"
                )
        object.__setattr__(self, "runtime_mode", mode)
        object.__setattr__(self, "endpoint_source", endpoint_source)
        object.__setattr__(self, "embedding_provider", provider)
        object.__setattr__(self, "vector_store", store)
        for name, value in text.items():
            object.__setattr__(self, name, value)

    def _validate_numbers(self) -> None:
        for name, value in (
            ("embedding_batch_size", self.embedding_batch_size),
            ("manifest_conflict_retry_limit", self.manifest_conflict_retry_limit),
            ("maximum_descriptor_batch_size", self.maximum_descriptor_batch_size),
            ("maximum_chunks_per_invocation", self.maximum_chunks_per_invocation),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise IndexingConfigurationError(f"{name} must be positive")
        if (
            isinstance(self.retry_limit, bool)
            or not isinstance(self.retry_limit, int)
            or self.retry_limit < 0
        ):
            raise IndexingConfigurationError("retry_limit cannot be negative")
        if self.embedding_dimensions is not None and (
            isinstance(self.embedding_dimensions, bool)
            or not isinstance(self.embedding_dimensions, int)
            or self.embedding_dimensions <= 0
        ):
            raise IndexingConfigurationError(
                "embedding_dimensions must be positive"
            )
        for name, value in (
            ("connect_timeout_seconds", self.connect_timeout_seconds),
            ("request_timeout_seconds", self.request_timeout_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise IndexingConfigurationError(f"{name} must be positive")

    def _normalized_text(self) -> dict[str, str | None]:
        required = {
            "embedding_model_id": self.embedding_model_id,
            "aws_region": self.aws_region,
            "ollama_url": self.ollama_url,
            "ollama_embedding_model": self.ollama_embedding_model,
            "qdrant_collection": self.qdrant_collection,
            "knowledge_namespace": self.knowledge_namespace,
            "knowledge_domain": self.knowledge_domain,
        }
        normalized: dict[str, str | None] = {}
        for name, value in required.items():
            if not isinstance(value, str) or not value.strip():
                raise IndexingConfigurationError(f"{name} cannot be empty")
            normalized[name] = value.strip()
        for name, value in (
            ("qdrant_url", self.qdrant_url),
            ("qdrant_api_key", self.qdrant_api_key),
            ("qdrant_secret_identifier", self.qdrant_secret_identifier),
            ("client_id", self.client_id),
            ("environment", self.environment),
        ):
            normalized[name] = (
                value.strip() if isinstance(value, str) and value.strip() else None
            )
        return normalized

    def _validate_scope(self, text: Mapping[str, str | None]) -> None:
        if text["client_id"] is None or text["environment"] is None:
            raise IndexingConfigurationError(
                "Enabled indexing requires client_id and environment"
            )

    def _validate_qdrant(
        self,
        mode: IndexingRuntimeMode,
        endpoint_source: VectorEndpointSource,
        text: Mapping[str, str | None],
    ) -> None:
        endpoint = text["qdrant_url"]
        secret_id = text["qdrant_secret_identifier"]
        if self.authentication_required and secret_id is None:
            raise IndexingConfigurationError(
                "Authenticated Qdrant requires a secret identifier"
            )
        if endpoint_source == VectorEndpointSource.ENVIRONMENT:
            if endpoint is None:
                raise IndexingConfigurationError(
                    "Environment endpoint source requires qdrant_url"
                )
            validate_qdrant_endpoint(
                endpoint,
                production=mode == IndexingRuntimeMode.PRODUCTION,
                tls_required=self.tls_required,
            )
        elif secret_id is None:
            raise IndexingConfigurationError(
                "Secret endpoint source requires a secret identifier"
            )

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str],
    ) -> "AutomaticIndexingConfig":
        """Read only the documented KNOWLEDGE_* configuration surface."""

        enabled = _boolean(
            environment.get("KNOWLEDGE_AUTOMATIC_INDEXING_ENABLED", "false"),
            "KNOWLEDGE_AUTOMATIC_INDEXING_ENABLED",
        )
        dimensions = environment.get("KNOWLEDGE_EMBEDDING_DIMENSIONS")
        return cls(
            enabled=enabled,
            runtime_mode=_required_enum(
                environment.get("KNOWLEDGE_INDEXING_RUNTIME_MODE", "local"),
                IndexingRuntimeMode,
                "KNOWLEDGE_INDEXING_RUNTIME_MODE",
            ),
            embedding_provider=_optional_enum(
                environment.get("KNOWLEDGE_EMBEDDING_PROVIDER"),
                IndexingEmbeddingProviderName,
            ),
            vector_store=_optional_enum(
                environment.get("KNOWLEDGE_VECTOR_STORE_PROVIDER"),
                IndexingVectorStoreName,
            ),
            embedding_model_id=environment.get(
                "KNOWLEDGE_EMBEDDING_MODEL_ID",
                "amazon.titan-embed-text-v2:0",
            ),
            embedding_batch_size=_positive_integer(
                environment.get("KNOWLEDGE_EMBEDDING_BATCH_SIZE", "8"),
                "KNOWLEDGE_EMBEDDING_BATCH_SIZE",
            ),
            embedding_dimensions=(
                _positive_integer(dimensions, "KNOWLEDGE_EMBEDDING_DIMENSIONS")
                if dimensions is not None and dimensions.strip()
                else None
            ),
            aws_region=environment.get("AWS_REGION", "us-west-2"),
            ollama_url=environment.get(
                "KNOWLEDGE_OLLAMA_URL", "http://localhost:11434"
            ),
            ollama_embedding_model=environment.get(
                "KNOWLEDGE_OLLAMA_EMBEDDING_MODEL", "embeddinggemma"
            ),
            qdrant_url=environment.get("KNOWLEDGE_QDRANT_URL"),
            qdrant_collection=environment.get(
                "KNOWLEDGE_QDRANT_COLLECTION",
                "dea_knowledge_embeddinggemma_v1",
            ),
            qdrant_api_key=environment.get("KNOWLEDGE_QDRANT_API_KEY"),
            qdrant_secret_identifier=environment.get(
                "KNOWLEDGE_QDRANT_SECRET_IDENTIFIER"
            ),
            endpoint_source=_required_enum(
                environment.get(
                    "KNOWLEDGE_QDRANT_ENDPOINT_SOURCE", "environment"
                ),
                VectorEndpointSource,
                "KNOWLEDGE_QDRANT_ENDPOINT_SOURCE",
            ),
            tls_required=_boolean(
                environment.get("KNOWLEDGE_QDRANT_TLS_REQUIRED", "false"),
                "KNOWLEDGE_QDRANT_TLS_REQUIRED",
            ),
            authentication_required=_boolean(
                environment.get(
                    "KNOWLEDGE_QDRANT_AUTHENTICATION_REQUIRED", "false"
                ),
                "KNOWLEDGE_QDRANT_AUTHENTICATION_REQUIRED",
            ),
            connect_timeout_seconds=_positive_float(
                environment.get("KNOWLEDGE_CONNECT_TIMEOUT_SECONDS", "5"),
                "KNOWLEDGE_CONNECT_TIMEOUT_SECONDS",
            ),
            request_timeout_seconds=_positive_float(
                environment.get("KNOWLEDGE_REQUEST_TIMEOUT_SECONDS", "10"),
                "KNOWLEDGE_REQUEST_TIMEOUT_SECONDS",
            ),
            retry_limit=_non_negative_integer(
                environment.get("KNOWLEDGE_INDEXING_RETRY_LIMIT", "2"),
                "KNOWLEDGE_INDEXING_RETRY_LIMIT",
            ),
            manifest_conflict_retry_limit=_positive_integer(
                environment.get("KNOWLEDGE_MANIFEST_CONFLICT_RETRIES", "3"),
                "KNOWLEDGE_MANIFEST_CONFLICT_RETRIES",
            ),
            maximum_descriptor_batch_size=_positive_integer(
                environment.get("KNOWLEDGE_MAX_DESCRIPTOR_BATCH_SIZE", "10"),
                "KNOWLEDGE_MAX_DESCRIPTOR_BATCH_SIZE",
            ),
            maximum_chunks_per_invocation=_positive_integer(
                environment.get("KNOWLEDGE_MAX_CHUNKS_PER_INVOCATION", "500"),
                "KNOWLEDGE_MAX_CHUNKS_PER_INVOCATION",
            ),
            client_id=environment.get("CLIENT_ID"),
            environment=environment.get("DEPLOYMENT_ENVIRONMENT"),
            knowledge_namespace=environment.get("KNOWLEDGE_NAMESPACE", "default"),
            knowledge_domain=environment.get("KNOWLEDGE_DOMAIN", "general"),
        )


class LazyAuthenticatedQdrantVectorStore:
    """Resolve credentials and construct Qdrant only on the first operation."""

    provider_name = "qdrant"

    def __init__(
        self,
        config: AutomaticIndexingConfig,
        resolver: QdrantCredentialResolver | None,
        *,
        store_factory: Callable[..., VectorStore] = QdrantVectorStore,
    ) -> None:
        self.collection_name = config.qdrant_collection
        self._config = config
        self._resolver = resolver
        self._store_factory = store_factory
        self._store: VectorStore | None = None

    def upsert(
        self,
        entries: Sequence[RetrievalEntry],
        *,
        client_id: str,
        environment: str,
    ) -> int:
        for entry in entries:
            require_indexable_metadata(
                entry.metadata,
                stage="vector_store_upsert",
            )
        return self._retry(
            lambda: self._get_store().upsert(
                entries, client_id=client_id, environment=environment
            )
        )

    def retrieve(
        self,
        query_vector: Sequence[float],
        *,
        client_id: str,
        environment: str,
        filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
        minimum_similarity: float | None = None,
    ) -> list[RetrievalResult]:
        return self._retry(
            lambda: self._get_store().retrieve(
                query_vector,
                client_id=client_id,
                environment=environment,
                filters=filters,
                top_k=top_k,
                minimum_similarity=minimum_similarity,
            )
        )

    def _get_store(self) -> VectorStore:
        if self._store is not None:
            return self._store
        endpoint = self._config.qdrant_url
        api_key = self._config.qdrant_api_key
        if self._resolver is not None:
            credentials = self._resolver.resolve()
            api_key = credentials.api_key
            if self._config.endpoint_source == VectorEndpointSource.SECRET:
                endpoint = credentials.endpoint
        if endpoint is None:
            raise IndexingConfigurationError(
                "Configured Qdrant endpoint could not be resolved"
            )
        validate_qdrant_endpoint(
            endpoint,
            production=(
                self._config.runtime_mode == IndexingRuntimeMode.PRODUCTION
            ),
            tls_required=self._config.tls_required,
        )
        try:
            self._store = self._store_factory(
                url=endpoint,
                collection_name=self._config.qdrant_collection,
                api_key=api_key,
                timeout_seconds=self._config.request_timeout_seconds,
            )
        except ImportError as error:
            raise IndexingDependencyError(
                "The configured vector-store dependency is unavailable"
            ) from error
        return self._store

    def _retry(self, operation: Callable[[], Any]) -> Any:
        last_error: Exception | None = None
        for _ in range(self._config.retry_limit + 1):
            try:
                return operation()
            except (IndexingConfigurationError, ValueError):
                raise
            except Exception as error:
                last_error = error
        raise IndexingDependencyError(
            "The vector-store operation failed after bounded retries"
        ) from last_error


@dataclass(frozen=True)
class IndexingReadinessReport:
    """Configuration-only readiness without connectivity probes."""

    ready: bool
    enabled: bool
    checks: tuple[str, ...]


def build_automatic_indexing_workflow(
    config: AutomaticIndexingConfig,
    *,
    storage: KnowledgeStorage,
    manifest: KnowledgeManifestRepository,
    bedrock_runtime_client: Any | None = None,
    ollama_http_session: Any | None = None,
    qdrant_client: Any | None = None,
    qdrant_models_module: Any | None = None,
    credential_resolver: QdrantCredentialResolver | None = None,
    qdrant_store_factory: Callable[..., VectorStore] | None = None,
) -> VectorIndexingWorkflow | None:
    """Compose dependencies without resolving secrets or making live calls."""

    if not config.enabled:
        return None
    if config.embedding_provider == IndexingEmbeddingProviderName.FAKE:
        provider = DeterministicFakeEmbeddingProvider(
            model_id=config.embedding_model_id,
            dimensions=config.embedding_dimensions or 16,
        )
    elif config.embedding_provider == IndexingEmbeddingProviderName.OLLAMA:
        provider = OllamaEmbeddingProvider(
            base_url=config.ollama_url,
            model_id=config.ollama_embedding_model,
            http_session=ollama_http_session,
        )
    elif config.embedding_provider == IndexingEmbeddingProviderName.BEDROCK:
        provider = BedrockEmbeddingProvider(
            model_id=config.embedding_model_id,
            region_name=config.aws_region,
            dimensions=config.embedding_dimensions,
            bedrock_runtime_client=bedrock_runtime_client,
        )
    else:  # pragma: no cover
        raise IndexingConfigurationError("Unsupported embedding provider")

    vector_store: VectorStore
    if config.vector_store == IndexingVectorStoreName.MEMORY:
        vector_store = InMemoryVectorStore()
    elif config.vector_store == IndexingVectorStoreName.QDRANT:
        if qdrant_client is not None:
            vector_store = QdrantVectorStore(
                url=config.qdrant_url or "https://configured-at-runtime.invalid",
                collection_name=config.qdrant_collection,
                api_key=config.qdrant_api_key or "injected-client",
                timeout_seconds=config.request_timeout_seconds,
                client=qdrant_client,
                models_module=qdrant_models_module,
            )
        else:
            resolver = credential_resolver
            if resolver is None and config.qdrant_secret_identifier is not None:
                resolver = SecretsManagerQdrantCredentialResolver(
                    config.qdrant_secret_identifier
                )
            vector_store = LazyAuthenticatedQdrantVectorStore(
                config,
                resolver,
                store_factory=qdrant_store_factory or QdrantVectorStore,
            )
    else:  # pragma: no cover
        raise IndexingConfigurationError("Unsupported vector store")

    return VectorIndexingWorkflow(
        storage=storage,
        embedding_workflow=EmbeddingWorkflow(
            storage=storage,
            provider=provider,
            model_id=provider.model_id,
            batch_size=config.embedding_batch_size,
            manifest=manifest,
        ),
        vector_store=vector_store,
        manifest=manifest,
        maximum_descriptor_batch_size=config.maximum_descriptor_batch_size,
        maximum_chunks_per_invocation=config.maximum_chunks_per_invocation,
    )


def check_indexing_readiness(
    config: AutomaticIndexingConfig,
    *,
    storage: KnowledgeStorage | None = None,
    manifest: KnowledgeManifestRepository | None = None,
) -> IndexingReadinessReport:
    """Validate composition without resolving secrets or probing providers."""

    if not config.enabled:
        return IndexingReadinessReport(
            ready=True,
            enabled=False,
            checks=("automatic_indexing_disabled",),
        )
    checks = [
        "configuration_valid",
        "client_scope_complete",
        "provider_supported",
        "vector_store_supported",
    ]
    if config.qdrant_secret_identifier:
        checks.append("secret_reference_present")
    if config.tls_required:
        checks.append("tls_required")
    if storage is not None and manifest is not None:
        build_automatic_indexing_workflow(
            config,
            storage=storage,
            manifest=manifest,
        )
        checks.append("dependency_composition_succeeded")
    return IndexingReadinessReport(
        ready=True,
        enabled=True,
        checks=tuple(checks),
    )


def validate_qdrant_endpoint(
    endpoint: str,
    *,
    production: bool,
    tls_required: bool,
) -> str:
    """Validate endpoint syntax and transport without making a request."""

    if not isinstance(endpoint, str):
        raise IndexingConfigurationError("Qdrant endpoint must be a string")
    normalized = endpoint.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise IndexingConfigurationError(
            "Qdrant endpoint must be an absolute HTTP or HTTPS URL"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise IndexingConfigurationError(
            "Qdrant endpoint must not contain credentials or query data"
        )
    is_loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme == "http" and (production or tls_required or not is_loopback):
        raise IndexingConfigurationError(
            "Insecure Qdrant endpoints are allowed only for explicit local loopback"
        )
    return normalized


def _optional_enum(value: str | None, enum_type: Any) -> Any:
    if not value or not value.strip():
        return None
    return _required_enum(value, enum_type, "provider")


def _required_enum(value: str, enum_type: Any, name: str) -> Any:
    try:
        return enum_type(value.strip().lower())
    except ValueError as error:
        raise IndexingConfigurationError(f"{name} is unsupported") from error


def _boolean(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise IndexingConfigurationError(f"{name} must be a boolean")


def _positive_integer(value: str, name: str) -> int:
    parsed = _integer(value, name)
    if parsed <= 0:
        raise IndexingConfigurationError(f"{name} must be greater than zero")
    return parsed


def _non_negative_integer(value: str, name: str) -> int:
    parsed = _integer(value, name)
    if parsed < 0:
        raise IndexingConfigurationError(f"{name} cannot be negative")
    return parsed


def _integer(value: str, name: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise IndexingConfigurationError(f"{name} must be an integer") from error


def _positive_float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise IndexingConfigurationError(f"{name} must be numeric") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise IndexingConfigurationError(f"{name} must be positive")
    return parsed
