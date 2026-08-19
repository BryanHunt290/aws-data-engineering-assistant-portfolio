"""Configuration for knowledge ingestion and chunking."""

from dataclasses import dataclass, field
import math
import re

from knowledge.intents import Intent


DEFAULT_SUPPORTED_DOCUMENT_TYPES = frozenset(
    {
        "docx",
        "html",
        "json",
        "md",
        "markdown",
        "pdf",
        "py",
        "txt",
    }
)


@dataclass(frozen=True)
class KnowledgeConfig:
    """Validated settings for the knowledge ingestion pipeline."""

    chunk_size: int = 1_000
    overlap: int = 100
    supported_document_types: frozenset[str] = field(
        default_factory=lambda: DEFAULT_SUPPORTED_DOCUMENT_TYPES
    )
    maximum_upload_size: int = 10 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.overlap < 0:
            raise ValueError("overlap cannot be negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        if self.maximum_upload_size <= 0:
            raise ValueError("maximum_upload_size must be greater than zero")

        normalized_types = frozenset(
            value.strip().lower().lstrip(".")
            for value in self.supported_document_types
            if value.strip().lstrip(".")
        )
        if not normalized_types:
            raise ValueError("supported_document_types cannot be empty")
        object.__setattr__(
            self,
            "supported_document_types",
            normalized_types,
        )


@dataclass(frozen=True)
class EmbeddingRetrievalConfig:
    """Validated settings for embedding and local retrieval services."""

    bedrock_region: str = "us-west-2"
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_batch_size: int = 8
    top_k: int = 5
    minimum_similarity_threshold: float = 0.0

    def __post_init__(self) -> None:
        region = self.bedrock_region.strip().lower()
        if not re.fullmatch(r"[a-z]{2}(?:-gov)?-[a-z]+-\d", region):
            raise ValueError("bedrock_region must be a valid AWS Region")
        model_id = self.embedding_model_id.strip()
        if not model_id:
            raise ValueError("embedding_model_id cannot be empty")
        if self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be greater than zero")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not -1.0 <= self.minimum_similarity_threshold <= 1.0:
            raise ValueError(
                "minimum_similarity_threshold must be between -1 and 1"
            )

        object.__setattr__(self, "bedrock_region", region)
        object.__setattr__(self, "embedding_model_id", model_id)


@dataclass(frozen=True)
class ClassificationRoutingConfig:
    """Validated intent classification and request-routing settings."""

    minimum_confidence: float = 0.60
    unknown_threshold: float = 0.45
    default_retrieval_top_k: int = 5
    approval_required_intents: frozenset[Intent] = field(
        default_factory=lambda: frozenset(
            {
                Intent.DEPLOYMENT_REQUEST,
                Intent.DESTRUCTIVE_ACTION_REQUEST,
            }
        )
    )
    safety_review_intents: frozenset[Intent] = field(
        default_factory=lambda: frozenset(
            {Intent.DESTRUCTIVE_ACTION_REQUEST}
        )
    )
    classifier_version: str = "rules-v1"

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence must be between zero and one"
            )
        if not 0.0 <= self.unknown_threshold <= 1.0:
            raise ValueError(
                "unknown_threshold must be between zero and one"
            )
        if self.unknown_threshold > self.minimum_confidence:
            raise ValueError(
                "unknown_threshold cannot exceed minimum_confidence"
            )
        if self.default_retrieval_top_k <= 0:
            raise ValueError(
                "default_retrieval_top_k must be greater than zero"
            )
        version = self.classifier_version.strip()
        if not version:
            raise ValueError("classifier_version cannot be empty")
        try:
            approval_intents = frozenset(
                Intent(value) for value in self.approval_required_intents
            )
            safety_intents = frozenset(
                Intent(value) for value in self.safety_review_intents
            )
        except ValueError as error:
            raise ValueError("Configured intents must be supported") from error
        required_approval = {
            Intent.DEPLOYMENT_REQUEST,
            Intent.DESTRUCTIVE_ACTION_REQUEST,
        }
        if not required_approval.issubset(approval_intents):
            raise ValueError(
                "Deployment and destructive intents must require approval"
            )
        if Intent.DESTRUCTIVE_ACTION_REQUEST not in safety_intents:
            raise ValueError(
                "Destructive actions must require safety review"
            )
        object.__setattr__(
            self,
            "approval_required_intents",
            approval_intents,
        )
        object.__setattr__(
            self,
            "safety_review_intents",
            safety_intents,
        )
        object.__setattr__(self, "classifier_version", version)


@dataclass(frozen=True)
class ApplicationConfig:
    """Validated settings for end-to-end RAG application orchestration."""

    bedrock_llm_region: str = "us-west-2"
    bedrock_llm_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    temperature: float = 0.1
    maximum_tokens: int = 1_024
    timeout_seconds: float = 30.0
    query_length_limit: int = 8_000
    context_length_limit: int = 24_000
    maximum_conversation_messages: int = 10
    maximum_retrieved_chunks: int = 5
    minimum_similarity: float = 0.0
    prompt_version: str = "grounded-rag-v1"
    application_version: str = "rag-application-v1"

    def __post_init__(self) -> None:
        region = self.bedrock_llm_region.strip().lower()
        if not re.fullmatch(r"[a-z]{2}(?:-gov)?-[a-z]+-\d", region):
            raise ValueError(
                "bedrock_llm_region must be a valid AWS Region"
            )
        model_id = self.bedrock_llm_model_id.strip()
        if not model_id:
            raise ValueError("bedrock_llm_model_id cannot be empty")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not 0.0 <= float(self.temperature) <= 1.0
        ):
            raise ValueError("temperature must be between zero and one")
        if (
            isinstance(self.maximum_tokens, bool)
            or not isinstance(self.maximum_tokens, int)
            or self.maximum_tokens <= 0
        ):
            raise ValueError("maximum_tokens must be greater than zero")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be greater than zero")
        if (
            isinstance(self.query_length_limit, bool)
            or not isinstance(self.query_length_limit, int)
            or self.query_length_limit <= 0
        ):
            raise ValueError("query_length_limit must be greater than zero")
        if (
            isinstance(self.context_length_limit, bool)
            or not isinstance(self.context_length_limit, int)
            or self.context_length_limit <= 0
        ):
            raise ValueError(
                "context_length_limit must be greater than zero"
            )
        if (
            isinstance(self.maximum_conversation_messages, bool)
            or not isinstance(self.maximum_conversation_messages, int)
            or self.maximum_conversation_messages < 0
        ):
            raise ValueError(
                "maximum_conversation_messages cannot be negative"
            )
        if (
            isinstance(self.maximum_retrieved_chunks, bool)
            or not isinstance(self.maximum_retrieved_chunks, int)
            or self.maximum_retrieved_chunks <= 0
        ):
            raise ValueError(
                "maximum_retrieved_chunks must be greater than zero"
            )
        if not -1.0 <= self.minimum_similarity <= 1.0:
            raise ValueError(
                "minimum_similarity must be between -1 and 1"
            )
        prompt_version = self.prompt_version.strip()
        application_version = self.application_version.strip()
        if not prompt_version:
            raise ValueError("prompt_version cannot be empty")
        if not application_version:
            raise ValueError("application_version cannot be empty")

        object.__setattr__(self, "bedrock_llm_region", region)
        object.__setattr__(self, "bedrock_llm_model_id", model_id)
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(
            self,
            "timeout_seconds",
            float(self.timeout_seconds),
        )
        object.__setattr__(self, "prompt_version", prompt_version)
        object.__setattr__(
            self,
            "application_version",
            application_version,
        )
