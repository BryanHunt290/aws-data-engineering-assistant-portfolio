"""Typed request, response, conversation, and attribution models."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import math
import re
from typing import Any, Mapping, Sequence

from knowledge.config import ApplicationConfig
from knowledge.costs import CostEstimate
from knowledge.intents import Intent
from knowledge.routing import Route


VALID_APPLICATION_ENVIRONMENTS = frozenset(
    {"dev", "test", "stage", "prod"}
)


class ConversationRole(StrEnum):
    """Roles retained in bounded conversation context."""

    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class ConversationMessage:
    """One prior message with optional scope evidence."""

    role: ConversationRole
    content: str
    client_id: str | None = None
    environment: str | None = None

    def __post_init__(self) -> None:
        try:
            role = ConversationRole(self.role)
        except ValueError as error:
            raise ValueError("Conversation message role is invalid") from error
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("Conversation message content cannot be empty")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "content", self.content.strip())


@dataclass(frozen=True)
class ApplicationRequest:
    """Validated request accepted by the RAG application service."""

    request_id: str
    query: str
    client_id: str
    environment: str
    conversation_context: tuple[ConversationMessage, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str):
            raise ValueError("request_id is invalid")
        if not isinstance(self.query, str):
            raise ValueError("query cannot be empty")
        if not isinstance(self.client_id, str):
            raise ValueError("client_id is invalid")
        if not isinstance(self.environment, str):
            raise ValueError("environment is invalid")
        request_id = self.request_id.strip()
        query = self.query.strip()
        client_id = self.client_id.strip().lower()
        environment = self.environment.strip().lower()
        if not request_id or len(request_id) > 128 or not re.fullmatch(
            r"[A-Za-z0-9._:-]+",
            request_id,
        ):
            raise ValueError("request_id is invalid")
        if not query:
            raise ValueError("query cannot be empty")
        if not re.fullmatch(
            r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?",
            client_id,
        ):
            raise ValueError("client_id is invalid")
        if environment not in VALID_APPLICATION_ENVIRONMENTS:
            raise ValueError("environment is invalid")
        if (
            not isinstance(self.timestamp, datetime)
            or self.timestamp.tzinfo is None
        ):
            raise ValueError("timestamp must include timezone information")

        messages = tuple(self.conversation_context)
        for message in messages:
            if not isinstance(message, ConversationMessage):
                raise ValueError(
                    "conversation_context must contain typed messages"
                )
            if message.client_id not in {None, client_id}:
                raise ValueError(
                    "conversation_context cannot mix client scopes"
                )
            if message.environment not in {None, environment}:
                raise ValueError(
                    "conversation_context cannot mix environments"
                )

        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "client_id", client_id)
        object.__setattr__(self, "environment", environment)
        object.__setattr__(self, "conversation_context", messages)
        try:
            metadata = dict(self.metadata)
        except (TypeError, ValueError) as error:
            raise ValueError("metadata must be a mapping") from error
        object.__setattr__(self, "metadata", metadata)

    def validate(self, config: ApplicationConfig) -> None:
        """Validate limits that belong to deploy-time application settings."""

        if len(self.query) > config.query_length_limit:
            raise ValueError("query exceeds query_length_limit")

    def bounded_conversation(
        self,
        config: ApplicationConfig,
    ) -> tuple[tuple[ConversationMessage, ...], bool]:
        """Return newest messages within count and character limits."""

        if config.maximum_conversation_messages == 0:
            return (), bool(self.conversation_context)
        selected = list(
            self.conversation_context[
                -config.maximum_conversation_messages :
            ]
        )
        was_truncated = len(selected) != len(self.conversation_context)
        remaining = config.context_length_limit
        bounded_reversed: list[ConversationMessage] = []
        for message in reversed(selected):
            if remaining <= 0:
                was_truncated = True
                break
            content = message.content
            if len(content) > remaining:
                content = content[-remaining:]
                was_truncated = True
            bounded_reversed.append(
                ConversationMessage(
                    role=message.role,
                    content=content,
                    client_id=message.client_id,
                    environment=message.environment,
                )
            )
            remaining -= len(content)
        return tuple(reversed(bounded_reversed)), was_truncated


class ApplicationStatus(StrEnum):
    """Terminal application outcomes."""

    COMPLETED = "completed"
    APPROVAL_REQUIRED = "approval_required"
    SAFETY_REVIEW_REQUIRED = "safety_review_required"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    FAILED = "failed"


@dataclass(frozen=True)
class SourceCitation:
    """Application-owned attribution independent of model-generated text."""

    source_id: str
    document_id: str
    chunk_id: str
    source_name: str
    object_key: str
    similarity_score: float
    page: str | int | None = None
    section: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id cannot be empty")
        if (
            not math.isfinite(self.similarity_score)
            or not -1.0 <= self.similarity_score <= 1.0
        ):
            raise ValueError(
                "similarity_score must be finite and between -1 and 1"
            )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class RetrievalMetadata:
    """Safe retrieval diagnostics returned to the caller."""

    attempted: bool
    result_count: int
    requested_top_k: int | None
    minimum_similarity: float | None
    context_characters: int
    filtered_for_scope: int = 0
    deduplicated: int = 0


@dataclass(frozen=True)
class ModelMetadata:
    """Safe language-model generation metadata."""

    provider_name: str | None
    model_id: str | None
    input_token_count: int | None
    output_token_count: int | None
    finish_reason: str | None
    latency_ms: float | None
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)
    cost_estimate: CostEstimate | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_metadata",
            dict(self.provider_metadata),
        )


EMPTY_MODEL_METADATA = ModelMetadata(
    provider_name=None,
    model_id=None,
    input_token_count=None,
    output_token_count=None,
    finish_reason=None,
    latency_ms=None,
)


@dataclass(frozen=True)
class ApplicationResponse:
    """Complete end-to-end application response."""

    request_id: str
    answer: str
    intent: Intent
    route: Route
    confidence: float
    sources: tuple[SourceCitation, ...]
    retrieval_metadata: RetrievalMetadata
    model_metadata: ModelMetadata
    approval_required: bool
    safety_review_required: bool
    latency_ms: float
    warnings: tuple[str, ...]
    status: ApplicationStatus
    error_category: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("latency_ms must be finite and non-negative")
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "warnings", tuple(self.warnings))
