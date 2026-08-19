"""Environment-backed, validated configuration for the local interface."""

from dataclasses import dataclass
from enum import StrEnum
import math
import re
from typing import Mapping


class RuntimeMode(StrEnum):
    """Explicit local runtime modes."""

    DEMO = "demo"
    BEDROCK = "bedrock"
    LOCAL = "local"


class LLMProviderName(StrEnum):
    """Supported language-model dependency selections."""

    FAKE = "fake"
    BEDROCK = "bedrock"
    OLLAMA = "ollama"


class EmbeddingProviderName(StrEnum):
    """Supported embedding dependency selections."""

    FAKE = "fake"
    BEDROCK = "bedrock"
    OLLAMA = "ollama"


class VectorStoreProviderName(StrEnum):
    """Supported vector-store dependency selections."""

    MEMORY = "memory"
    QDRANT = "qdrant"


VALID_UI_ENVIRONMENTS = frozenset({"dev", "test", "stage", "prod"})


@dataclass(frozen=True)
class UIConfig:
    """Configuration consumed by UI bootstrap and presentation code."""

    runtime_mode: RuntimeMode = RuntimeMode.DEMO
    llm_provider: LLMProviderName = LLMProviderName.FAKE
    embedding_provider: EmbeddingProviderName = EmbeddingProviderName.FAKE
    vector_store_provider: VectorStoreProviderName = (
        VectorStoreProviderName.MEMORY
    )
    aws_region: str = "us-west-2"
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    llm_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    default_client_id: str = "demo-client"
    default_environment: str = "dev"
    retrieval_top_k: int = 5
    minimum_similarity: float = 0.0
    maximum_conversation_messages: int = 10
    developer_mode: bool = False
    pricing_catalog_path: str | None = None
    ollama_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "embeddinggemma"
    ollama_chat_model: str = "qwen3:8b"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "dea_knowledge_embeddinggemma_v1"
    qdrant_api_key: str | None = None
    provider_selection_explicit: bool = False

    def __post_init__(self) -> None:
        try:
            mode = RuntimeMode(self.runtime_mode)
        except ValueError as error:
            raise ValueError(
                "runtime_mode must be demo, bedrock, or local"
            ) from error
        try:
            llm_provider = LLMProviderName(self.llm_provider)
            embedding_provider = EmbeddingProviderName(
                self.embedding_provider
            )
            vector_store_provider = VectorStoreProviderName(
                self.vector_store_provider
            )
        except ValueError as error:
            raise ValueError("Unsupported provider configuration") from error
        default_providers = (
            llm_provider == LLMProviderName.FAKE
            and embedding_provider == EmbeddingProviderName.FAKE
            and vector_store_provider == VectorStoreProviderName.MEMORY
        )
        if (
            default_providers
            and not self.provider_selection_explicit
            and mode == RuntimeMode.BEDROCK
        ):
            llm_provider = LLMProviderName.BEDROCK
            embedding_provider = EmbeddingProviderName.BEDROCK
        elif (
            default_providers
            and not self.provider_selection_explicit
            and mode == RuntimeMode.LOCAL
        ):
            llm_provider = LLMProviderName.OLLAMA
            embedding_provider = EmbeddingProviderName.OLLAMA
            vector_store_provider = VectorStoreProviderName.QDRANT
        region = self.aws_region.strip().lower()
        if not re.fullmatch(r"[a-z]{2}(?:-gov)?-[a-z]+-\d", region):
            raise ValueError("aws_region must be a valid AWS Region")
        embedding_model_id = self.embedding_model_id.strip()
        llm_model_id = self.llm_model_id.strip()
        client_id = self.default_client_id.strip().lower()
        environment = self.default_environment.strip().lower()
        if not embedding_model_id:
            raise ValueError("embedding_model_id cannot be empty")
        if not llm_model_id:
            raise ValueError("llm_model_id cannot be empty")
        if not re.fullmatch(
            r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?",
            client_id,
        ):
            raise ValueError("default_client_id is invalid")
        if environment not in VALID_UI_ENVIRONMENTS:
            raise ValueError("default_environment is invalid")
        if (
            isinstance(self.retrieval_top_k, bool)
            or not isinstance(self.retrieval_top_k, int)
            or not 1 <= self.retrieval_top_k <= 50
        ):
            raise ValueError("retrieval_top_k must be between 1 and 50")
        if (
            isinstance(self.minimum_similarity, bool)
            or not isinstance(self.minimum_similarity, (int, float))
            or not math.isfinite(float(self.minimum_similarity))
            or not -1.0 <= float(self.minimum_similarity) <= 1.0
        ):
            raise ValueError(
                "minimum_similarity must be between -1 and 1"
            )
        if (
            isinstance(self.maximum_conversation_messages, bool)
            or not isinstance(self.maximum_conversation_messages, int)
            or not 0 <= self.maximum_conversation_messages <= 100
        ):
            raise ValueError(
                "maximum_conversation_messages must be between 0 and 100"
            )
        if not isinstance(self.developer_mode, bool):
            raise ValueError("developer_mode must be a boolean")
        if not isinstance(self.provider_selection_explicit, bool):
            raise ValueError("provider_selection_explicit must be a boolean")
        catalog_path = (
            self.pricing_catalog_path.strip()
            if self.pricing_catalog_path
            else None
        )
        ollama_url = self.ollama_url.strip().rstrip("/")
        ollama_embedding_model = self.ollama_embedding_model.strip()
        ollama_chat_model = self.ollama_chat_model.strip()
        qdrant_url = self.qdrant_url.strip().rstrip("/")
        qdrant_collection = self.qdrant_collection.strip()
        qdrant_api_key = (
            self.qdrant_api_key.strip()
            if self.qdrant_api_key and self.qdrant_api_key.strip()
            else None
        )
        if (
            llm_provider == LLMProviderName.OLLAMA
            or embedding_provider == EmbeddingProviderName.OLLAMA
        ) and (
            not ollama_url
            or not ollama_embedding_model
            or not ollama_chat_model
        ):
            raise ValueError(
                "Ollama URL and configured model names cannot be empty"
            )
        if vector_store_provider == VectorStoreProviderName.QDRANT and (
            not qdrant_url or not qdrant_collection
        ):
            raise ValueError(
                "Qdrant URL and collection name cannot be empty"
            )

        object.__setattr__(self, "runtime_mode", mode)
        object.__setattr__(self, "llm_provider", llm_provider)
        object.__setattr__(
            self,
            "embedding_provider",
            embedding_provider,
        )
        object.__setattr__(
            self,
            "vector_store_provider",
            vector_store_provider,
        )
        object.__setattr__(self, "aws_region", region)
        object.__setattr__(
            self,
            "embedding_model_id",
            embedding_model_id,
        )
        object.__setattr__(self, "llm_model_id", llm_model_id)
        object.__setattr__(self, "default_client_id", client_id)
        object.__setattr__(self, "default_environment", environment)
        object.__setattr__(
            self,
            "minimum_similarity",
            float(self.minimum_similarity),
        )
        object.__setattr__(self, "pricing_catalog_path", catalog_path)
        object.__setattr__(self, "ollama_url", ollama_url)
        object.__setattr__(
            self,
            "ollama_embedding_model",
            ollama_embedding_model,
        )
        object.__setattr__(self, "ollama_chat_model", ollama_chat_model)
        object.__setattr__(self, "qdrant_url", qdrant_url)
        object.__setattr__(self, "qdrant_collection", qdrant_collection)
        object.__setattr__(self, "qdrant_api_key", qdrant_api_key)


_ENVIRONMENT_KEYS = {
    "runtime_mode": ("APP_RUNTIME_MODE", "DEA_RUNTIME_MODE"),
    "llm_provider": (
        "APP_LLM_PROVIDER",
        "LLM_PROVIDER",
        "DEA_LLM_PROVIDER",
    ),
    "embedding_provider": (
        "APP_EMBEDDING_PROVIDER",
        "EMBEDDING_PROVIDER",
        "DEA_EMBEDDING_PROVIDER",
    ),
    "vector_store_provider": (
        "APP_VECTOR_STORE_PROVIDER",
        "VECTOR_STORE_PROVIDER",
        "DEA_VECTOR_STORE_PROVIDER",
    ),
    "aws_region": ("AWS_REGION", "DEA_AWS_REGION"),
    "embedding_model_id": (
        "APP_EMBEDDING_MODEL_ID",
        "DEA_EMBEDDING_MODEL_ID",
    ),
    "llm_model_id": ("APP_LLM_MODEL_ID", "DEA_LLM_MODEL_ID"),
    "default_client_id": (
        "APP_DEFAULT_CLIENT_ID",
        "DEA_DEFAULT_CLIENT_ID",
    ),
    "default_environment": (
        "APP_DEFAULT_ENVIRONMENT",
        "DEA_DEFAULT_ENVIRONMENT",
    ),
    "retrieval_top_k": (
        "APP_RETRIEVAL_TOP_K",
        "DEA_RETRIEVAL_TOP_K",
    ),
    "minimum_similarity": (
        "APP_MINIMUM_SIMILARITY",
        "DEA_MINIMUM_SIMILARITY",
    ),
    "maximum_conversation_messages": (
        "APP_MAXIMUM_CONVERSATION_MESSAGES",
        "DEA_MAXIMUM_CONVERSATION_MESSAGES",
    ),
    "developer_mode": ("APP_DEVELOPER_MODE", "DEA_DEVELOPER_MODE"),
    "pricing_catalog_path": (
        "APP_PRICING_CATALOG_PATH",
        "DEA_PRICING_CATALOG_PATH",
    ),
    "ollama_url": (
        "APP_OLLAMA_URL",
        "OLLAMA_URL",
        "DEA_OLLAMA_BASE_URL",
    ),
    "ollama_embedding_model": (
        "APP_OLLAMA_EMBEDDING_MODEL",
        "OLLAMA_EMBEDDING_MODEL",
        "DEA_OLLAMA_EMBEDDING_MODEL",
    ),
    "ollama_chat_model": (
        "APP_OLLAMA_CHAT_MODEL",
        "OLLAMA_CHAT_MODEL",
        "DEA_OLLAMA_MODEL_ID",
    ),
    "qdrant_url": (
        "APP_QDRANT_URL",
        "QDRANT_URL",
        "DEA_QDRANT_URL",
    ),
    "qdrant_collection": (
        "APP_QDRANT_COLLECTION",
        "QDRANT_COLLECTION",
        "DEA_QDRANT_COLLECTION",
    ),
    "qdrant_api_key": (
        "APP_QDRANT_API_KEY",
        "QDRANT_API_KEY",
        "DEA_QDRANT_API_KEY",
    ),
}


def load_ui_config(
    environment: Mapping[str, str] | None = None,
    *,
    overrides: Mapping[str, object] | None = None,
) -> UIConfig:
    """Load environment variables plus optional non-secret local overrides."""

    if environment is None:
        import os

        environment = os.environ
    defaults = UIConfig()
    values: dict[str, object] = {
        field_name: getattr(defaults, field_name)
        for field_name in _ENVIRONMENT_KEYS
    }
    provider_environment_fields = {
        "llm_provider",
        "embedding_provider",
        "vector_store_provider",
    }
    provider_selection_explicit = False
    for field_name, environment_keys in _ENVIRONMENT_KEYS.items():
        environment_key, raw_value = _first_environment_value(
            environment,
            environment_keys,
        )
        if environment_key is None:
            continue
        if raw_value is None or not raw_value.strip():
            continue
        if field_name in provider_environment_fields:
            provider_selection_explicit = True
        if field_name in {
            "retrieval_top_k",
            "maximum_conversation_messages",
        }:
            values[field_name] = _parse_int(raw_value, environment_key)
        elif field_name == "minimum_similarity":
            values[field_name] = _parse_float(raw_value, environment_key)
        elif field_name == "developer_mode":
            values[field_name] = _parse_bool(raw_value, environment_key)
        else:
            values[field_name] = raw_value
    values.update(dict(overrides or {}))
    if any(
        key in dict(overrides or {}) for key in provider_environment_fields
    ):
        provider_selection_explicit = True
    values["provider_selection_explicit"] = provider_selection_explicit
    return UIConfig(**values)


def _first_environment_value(
    environment: Mapping[str, str],
    keys: tuple[str, ...],
) -> tuple[str | None, str | None]:
    for key in keys:
        value = environment.get(key)
        if value is not None and value.strip():
            return key, value
    return None, None


def _parse_int(value: str, name: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def _parse_float(value: str, name: str) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error


def _parse_bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")
