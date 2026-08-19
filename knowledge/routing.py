"""Request routing plans that never execute tools or mutate infrastructure."""

from dataclasses import dataclass
from enum import StrEnum
import json
import logging
from time import perf_counter

from knowledge.config import ClassificationRoutingConfig
from knowledge.intents import ClassificationResult, Intent, KnowledgeScope


class Route(StrEnum):
    """Supported downstream request-handling routes."""

    DIRECT_RESPONSE = "direct_response"
    RETRIEVAL = "retrieval"
    REQUIREMENTS_GATHERING = "requirements_gathering"
    CODE_GENERATION = "code_generation"
    TROUBLESHOOTING = "troubleshooting"
    TOOL_EXECUTION = "tool_execution"
    APPROVAL_REQUIRED = "approval_required"
    REJECTION_OR_SAFETY_REVIEW = "rejection_or_safety_review"


class ToolCategory(StrEnum):
    """Tool family a future orchestrator may consider after routing."""

    NONE = "none"
    CODE_GENERATION = "code_generation"
    DIAGNOSTICS = "diagnostics"
    MONITORING = "monitoring"
    DEPLOYMENT = "deployment"
    DESTRUCTIVE_OPERATION = "destructive_operation"


@dataclass(frozen=True)
class RoutingPlan:
    """A non-executing, client-scoped plan produced from a classification."""

    selected_route: Route
    intent: Intent
    retrieval_required: bool
    retrieval_top_k: int | None
    retrieval_scope: KnowledgeScope
    tool_category: ToolCategory
    approval_required: bool
    safety_review_required: bool
    next_action: str
    classifier_confidence: float
    client_id: str | None
    environment: str | None


_CODE_INTENTS = frozenset(
    {
        Intent.PIPELINE_GENERATION,
        Intent.SQL_GENERATION,
        Intent.PYSPARK_GENERATION,
        Intent.CDK_GENERATION,
    }
)

_TROUBLESHOOTING_INTENTS = frozenset(
    {
        Intent.PIPELINE_TROUBLESHOOTING,
        Intent.AWS_ERROR_EXPLANATION,
    }
)

_RETRIEVAL_INTENTS = frozenset(
    {
        Intent.ARCHITECTURE_DESIGN,
        Intent.DATA_QUALITY,
        Intent.COST_QUESTION,
        Intent.KNOWLEDGE_QUESTION,
        Intent.IAM_REVIEW,
    }
)


class RequestRouter:
    """Map a classification into a safe plan without executing it."""

    def __init__(
        self,
        config: ClassificationRoutingConfig | None = None,
        *,
        event_logger: logging.Logger | None = None,
    ) -> None:
        self._config = config or ClassificationRoutingConfig()
        self._logger = event_logger or logging.getLogger(__name__)

    def route(
        self,
        classification: ClassificationResult,
        *,
        client_id: str | None = None,
        environment: str | None = None,
    ) -> RoutingPlan:
        started = perf_counter()
        intent = classification.intent
        approval_required = (
            classification.requires_approval
            or intent in self._config.approval_required_intents
        )
        safety_review_required = (
            intent in self._config.safety_review_intents
        )

        if safety_review_required:
            selected_route = Route.REJECTION_OR_SAFETY_REVIEW
            tool_category = ToolCategory.DESTRUCTIVE_OPERATION
            next_action = (
                "Pause for explicit approval and a scoped safety review."
            )
        elif approval_required:
            selected_route = Route.APPROVAL_REQUIRED
            tool_category = ToolCategory.DEPLOYMENT
            next_action = "Pause for explicit approval before any tool call."
        elif intent == Intent.PIPELINE_REQUIREMENTS or intent == Intent.UNKNOWN:
            selected_route = Route.REQUIREMENTS_GATHERING
            tool_category = ToolCategory.NONE
            next_action = "Ask a focused clarifying question."
        elif intent in _CODE_INTENTS:
            selected_route = Route.CODE_GENERATION
            tool_category = ToolCategory.CODE_GENERATION
            next_action = "Prepare code within the confirmed request scope."
        elif intent in _TROUBLESHOOTING_INTENTS:
            selected_route = Route.TROUBLESHOOTING
            tool_category = ToolCategory.DIAGNOSTICS
            next_action = "Diagnose using available scoped evidence."
        elif classification.requires_tool_call:
            selected_route = Route.TOOL_EXECUTION
            tool_category = ToolCategory.MONITORING
            next_action = (
                "Prepare a read-only scoped tool request; do not execute yet."
            )
        elif classification.requires_retrieval or intent in _RETRIEVAL_INTENTS:
            selected_route = Route.RETRIEVAL
            tool_category = ToolCategory.NONE
            next_action = "Retrieve from the permitted knowledge scope."
        else:
            selected_route = Route.DIRECT_RESPONSE
            tool_category = ToolCategory.NONE
            next_action = "Respond directly without tools."

        retrieval_required = (
            selected_route == Route.RETRIEVAL
            or classification.requires_retrieval
        )
        retrieval_scope = (
            classification.preferred_knowledge_scope
            if retrieval_required
            else KnowledgeScope.NONE
        )
        plan = RoutingPlan(
            selected_route=selected_route,
            intent=intent,
            retrieval_required=retrieval_required,
            retrieval_top_k=(
                self._config.default_retrieval_top_k
                if retrieval_required
                else None
            ),
            retrieval_scope=retrieval_scope,
            tool_category=tool_category,
            approval_required=approval_required,
            safety_review_required=safety_review_required,
            next_action=next_action,
            classifier_confidence=classification.confidence,
            client_id=client_id,
            environment=environment,
        )
        self._emit(plan, perf_counter() - started)
        return plan

    def _emit(self, plan: RoutingPlan, elapsed_seconds: float) -> None:
        event = {
            "classifier_confidence": plan.classifier_confidence,
            "classifier_version": self._config.classifier_version,
            "client_id": plan.client_id,
            "elapsed_ms": round(elapsed_seconds * 1_000, 3),
            "environment": plan.environment,
            "event": "request_routing",
            "predicted_intent": plan.intent.value,
            "route": plan.selected_route.value,
        }
        self._logger.info(json.dumps(event, sort_keys=True))
