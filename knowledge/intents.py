"""Typed, provider-neutral intent-classification contracts."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Protocol, runtime_checkable


class Intent(StrEnum):
    """Supported user-request intents."""

    ARCHITECTURE_DESIGN = "architecture_design"
    PIPELINE_REQUIREMENTS = "pipeline_requirements"
    PIPELINE_GENERATION = "pipeline_generation"
    PIPELINE_TROUBLESHOOTING = "pipeline_troubleshooting"
    AWS_ERROR_EXPLANATION = "aws_error_explanation"
    SQL_GENERATION = "sql_generation"
    PYSPARK_GENERATION = "pyspark_generation"
    CDK_GENERATION = "cdk_generation"
    IAM_REVIEW = "iam_review"
    DATA_QUALITY = "data_quality"
    MONITORING_REQUEST = "monitoring_request"
    COST_QUESTION = "cost_question"
    KNOWLEDGE_QUESTION = "knowledge_question"
    DEPLOYMENT_REQUEST = "deployment_request"
    DESTRUCTIVE_ACTION_REQUEST = "destructive_action_request"
    GENERAL_CONVERSATION = "general_conversation"
    UNKNOWN = "unknown"


class KnowledgeScope(StrEnum):
    """Permitted knowledge boundaries requested by a classification."""

    NONE = "none"
    CLIENT = "client"
    CLIENT_ENVIRONMENT = "client_environment"
    GLOBAL = "global"


@dataclass(frozen=True)
class ClassificationResult:
    """A user-safe classification result returned by any provider."""

    intent: Intent
    confidence: float
    requires_retrieval: bool
    requires_tool_call: bool
    requires_approval: bool
    preferred_knowledge_scope: KnowledgeScope
    reasoning_summary: str
    matched_rules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        summary = self.reasoning_summary.strip()
        if not summary:
            raise ValueError("reasoning_summary cannot be empty")
        if len(summary) > 240:
            raise ValueError("reasoning_summary must be user-safe and short")
        object.__setattr__(self, "reasoning_summary", summary)
        object.__setattr__(self, "matched_rules", tuple(self.matched_rules))


@runtime_checkable
class IntentClassifier(Protocol):
    """Provider-neutral interface for request intent classification."""

    @property
    def classifier_version(self) -> str:
        """Return the stable classifier implementation version."""

    def classify(
        self,
        query: str,
        *,
        conversation_context: str | None = None,
        client_id: str | None = None,
        environment: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ClassificationResult:
        """Classify a request without executing it."""
