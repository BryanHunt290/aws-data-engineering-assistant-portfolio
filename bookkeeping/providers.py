"""Provider selection for explicit bookkeeping advisory operations."""

from typing import Any

from bookkeeping.config import BookkeepingConfig, BookkeepingLLMProvider
from knowledge.bedrock_llm import BedrockLLMProvider
from knowledge.fake_llm import DeterministicFakeLLMProvider
from knowledge.llm import LLMProvider
from knowledge.ollama_llm import OllamaLLMProvider


def build_bookkeeping_llm_provider(
    config: BookkeepingConfig,
    *,
    fake_provider: LLMProvider | None = None,
    http_session: Any | None = None,
    bedrock_runtime_client: Any | None = None,
) -> LLMProvider:
    """Build exactly the configured provider without fallback."""

    if config.llm_provider == BookkeepingLLMProvider.FAKE:
        return fake_provider or DeterministicFakeLLMProvider(
            response_text=(
                "Offline fake explanation. Review the deterministic metrics "
                "and supporting transaction references."
            ),
            model_id="bookkeeping-fake-v1",
            provider_metadata={"mode": "offline"},
        )
    if config.llm_provider == BookkeepingLLMProvider.OLLAMA:
        return OllamaLLMProvider(
            base_url=config.ollama_base_url,
            model_id=config.ollama_model,
            connect_timeout_seconds=(
                config.ollama_connect_timeout_seconds
            ),
            read_timeout_seconds=config.ollama_read_timeout_seconds,
            http_session=http_session,
        )
    return BedrockLLMProvider(
        model_id=config.bedrock_model_id,
        region_name=config.bedrock_region,
        timeout_seconds=config.ollama_read_timeout_seconds,
        bedrock_runtime_client=bedrock_runtime_client,
    )
