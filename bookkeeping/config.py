"""Validated environment configuration for local bookkeeping analysis."""

from dataclasses import dataclass
from enum import StrEnum
import math
import re
from typing import Mapping


class BookkeepingLLMProvider(StrEnum):
    """Explicit advisory language-model providers."""

    FAKE = "fake"
    OLLAMA = "ollama"
    BEDROCK = "bedrock"


@dataclass(frozen=True)
class BookkeepingConfig:
    """Limits and provider settings for the bookkeeping capability."""

    llm_provider: BookkeepingLLMProvider = BookkeepingLLMProvider.FAKE
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gpt-oss:20b"
    ollama_connect_timeout_seconds: float = 5.0
    ollama_read_timeout_seconds: float = 120.0
    bedrock_region: str = "us-west-2"
    bedrock_model_id: str = (
        "anthropic.claude-3-haiku-20240307-v1:0"
    )
    maximum_upload_size_bytes: int = 5 * 1024 * 1024
    maximum_rows: int = 10_000
    categorization_batch_size: int = 25
    duplicate_date_window_days: int = 3
    knowledge_maximum_passages: int = 5
    knowledge_maximum_context_characters: int = 12_000
    knowledge_chunk_size: int = 1_000
    knowledge_chunk_overlap: int = 100

    def __post_init__(self) -> None:
        try:
            provider = BookkeepingLLMProvider(self.llm_provider)
        except ValueError as error:
            raise ValueError(
                "llm_provider must be fake, ollama, or bedrock"
            ) from error
        base_url = self.ollama_base_url.strip().rstrip("/")
        model = self.ollama_model.strip()
        region = self.bedrock_region.strip().lower()
        bedrock_model = self.bedrock_model_id.strip()
        if not base_url:
            raise ValueError("ollama_base_url cannot be empty")
        if not model:
            raise ValueError("ollama_model cannot be empty")
        if not re.fullmatch(r"[a-z]{2}(?:-gov)?-[a-z]+-\d", region):
            raise ValueError("bedrock_region must be a valid AWS Region")
        if not bedrock_model:
            raise ValueError("bedrock_model_id cannot be empty")
        for name, value in (
            (
                "ollama_connect_timeout_seconds",
                self.ollama_connect_timeout_seconds,
            ),
            (
                "ollama_read_timeout_seconds",
                self.ollama_read_timeout_seconds,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive")
        for name, value in (
            ("maximum_upload_size_bytes", self.maximum_upload_size_bytes),
            ("maximum_rows", self.maximum_rows),
            ("categorization_batch_size", self.categorization_batch_size),
            ("knowledge_maximum_passages", self.knowledge_maximum_passages),
            (
                "knowledge_maximum_context_characters",
                self.knowledge_maximum_context_characters,
            ),
            ("knowledge_chunk_size", self.knowledge_chunk_size),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be greater than zero")
        if (
            isinstance(self.knowledge_chunk_overlap, bool)
            or not isinstance(self.knowledge_chunk_overlap, int)
            or self.knowledge_chunk_overlap < 0
            or self.knowledge_chunk_overlap >= self.knowledge_chunk_size
        ):
            raise ValueError(
                "knowledge_chunk_overlap must be non-negative and smaller "
                "than knowledge_chunk_size"
            )
        if (
            isinstance(self.duplicate_date_window_days, bool)
            or not isinstance(self.duplicate_date_window_days, int)
            or self.duplicate_date_window_days < 0
        ):
            raise ValueError(
                "duplicate_date_window_days cannot be negative"
            )

        object.__setattr__(self, "llm_provider", provider)
        object.__setattr__(self, "ollama_base_url", base_url)
        object.__setattr__(self, "ollama_model", model)
        object.__setattr__(
            self,
            "ollama_connect_timeout_seconds",
            float(self.ollama_connect_timeout_seconds),
        )
        object.__setattr__(
            self,
            "ollama_read_timeout_seconds",
            float(self.ollama_read_timeout_seconds),
        )
        object.__setattr__(self, "bedrock_region", region)
        object.__setattr__(self, "bedrock_model_id", bedrock_model)


def load_bookkeeping_config(
    environment: Mapping[str, str] | None = None,
) -> BookkeepingConfig:
    """Load bookkeeping configuration without reading secrets."""

    if environment is None:
        import os

        environment = os.environ
    defaults = BookkeepingConfig()
    return BookkeepingConfig(
        llm_provider=environment.get(
            "DEA_LLM_PROVIDER",
            defaults.llm_provider.value,
        ),
        ollama_base_url=environment.get(
            "DEA_OLLAMA_BASE_URL",
            defaults.ollama_base_url,
        ),
        ollama_model=environment.get(
            "DEA_OLLAMA_MODEL",
            defaults.ollama_model,
        ),
        ollama_connect_timeout_seconds=_environment_float(
            environment,
            "DEA_OLLAMA_CONNECT_TIMEOUT_SECONDS",
            defaults.ollama_connect_timeout_seconds,
        ),
        ollama_read_timeout_seconds=_environment_float(
            environment,
            "DEA_OLLAMA_TIMEOUT_SECONDS",
            defaults.ollama_read_timeout_seconds,
        ),
        bedrock_region=environment.get(
            "DEA_AWS_REGION",
            defaults.bedrock_region,
        ),
        bedrock_model_id=environment.get(
            "DEA_LLM_MODEL_ID",
            defaults.bedrock_model_id,
        ),
        maximum_upload_size_bytes=_environment_int(
            environment,
            "DEA_BOOKKEEPING_MAX_UPLOAD_BYTES",
            defaults.maximum_upload_size_bytes,
        ),
        maximum_rows=_environment_int(
            environment,
            "DEA_BOOKKEEPING_MAX_ROWS",
            defaults.maximum_rows,
        ),
        categorization_batch_size=_environment_int(
            environment,
            "DEA_BOOKKEEPING_CATEGORY_BATCH_SIZE",
            defaults.categorization_batch_size,
        ),
        duplicate_date_window_days=_environment_int(
            environment,
            "DEA_BOOKKEEPING_DUPLICATE_WINDOW_DAYS",
            defaults.duplicate_date_window_days,
        ),
        knowledge_maximum_passages=_environment_int(
            environment,
            "DEA_BOOKKEEPING_KNOWLEDGE_MAX_PASSAGES",
            defaults.knowledge_maximum_passages,
        ),
        knowledge_maximum_context_characters=_environment_int(
            environment,
            "DEA_BOOKKEEPING_KNOWLEDGE_MAX_CONTEXT_CHARACTERS",
            defaults.knowledge_maximum_context_characters,
        ),
        knowledge_chunk_size=_environment_int(
            environment,
            "DEA_BOOKKEEPING_KNOWLEDGE_CHUNK_SIZE",
            defaults.knowledge_chunk_size,
        ),
        knowledge_chunk_overlap=_environment_int(
            environment,
            "DEA_BOOKKEEPING_KNOWLEDGE_CHUNK_OVERLAP",
            defaults.knowledge_chunk_overlap,
        ),
    )


def _environment_int(
    environment: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def _environment_float(
    environment: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
