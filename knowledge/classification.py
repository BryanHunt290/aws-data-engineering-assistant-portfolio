"""Deterministic and future provider implementations for intent classification."""

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Mapping, Sequence

from knowledge.config import ClassificationRoutingConfig
from knowledge.intents import (
    ClassificationResult,
    Intent,
    KnowledgeScope,
)


@dataclass(frozen=True)
class IntentRule:
    """Configurable phrase and keyword rule with an explicit priority."""

    name: str
    intent: Intent
    priority: int
    phrases: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    minimum_keyword_matches: int = 2
    requires_retrieval: bool = False
    requires_tool_call: bool = False
    preferred_knowledge_scope: KnowledgeScope = (
        KnowledgeScope.CLIENT_ENVIRONMENT
    )

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Rule name cannot be empty")
        if self.priority < 0:
            raise ValueError("Rule priority cannot be negative")
        if not self.phrases and not self.keywords:
            raise ValueError("A rule must define phrases or keywords")
        if self.minimum_keyword_matches <= 0:
            raise ValueError("minimum_keyword_matches must be positive")


@dataclass(frozen=True)
class _RuleMatch:
    rule: IntentRule
    confidence: float


_DISCUSSION_PREFIXES = (
    "can you explain",
    "could you explain",
    "describe",
    "discuss",
    "explain",
    "how does",
    "how would",
    "is it safe",
    "i want to understand",
    "help me understand",
    "please describe",
    "please discuss",
    "please explain",
    "review whether",
    "should i",
    "should we",
    "tell me about",
    "what does",
    "what happens",
    "what is",
    "why does",
    "why is",
)

_DESTRUCTIVE_PATTERNS = (
    r"\b(delete|destroy|replace)\b",
    r"\boverwrite\b.*"
    r"\b(bucket|data|databases?|files?|objects?|resources?|tables?)\b",
    r"\bmove\b.*\b(data|files?|objects?|tables?)\b",
    r"\bmodify\b.*\b(prod|production)\b",
    r"\bbroaden\b.*\b(iam|permission|permissions|policy)\b",
)

_DEPLOYMENT_PATTERNS = (
    r"\bdeploy\b",
    r"\brelease\b.*\b(prod|production|stack|application|lambda)\b",
    r"\bapply\b.*\b(cdk|cloudformation|stack)\b",
)


def normalize_request_text(value: str) -> str:
    """Normalize case, Unicode, punctuation, and whitespace for matching."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[_/\\-]+", " ", normalized)
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return " ".join(normalized.split())


def _default_rules() -> tuple[IntentRule, ...]:
    return (
        IntentRule(
            "aws_error",
            Intent.AWS_ERROR_EXPLANATION,
            850,
            phrases=(
                "access denied error",
                "aws error",
                "explain this error",
                "iam error",
                "what does this error mean",
            ),
            keywords=("error", "exception", "failed", "access denied"),
            minimum_keyword_matches=2,
            requires_retrieval=True,
        ),
        IntentRule(
            "pipeline_troubleshooting",
            Intent.PIPELINE_TROUBLESHOOTING,
            840,
            phrases=(
                "glue job failed",
                "glue job fail",
                "glue job is failing",
                "pipeline failed",
                "pipeline is failing",
                "troubleshoot pipeline",
            ),
            keywords=(
                "debug",
                "failed",
                "fail",
                "failing",
                "glue job",
                "pipeline",
                "troubleshoot",
            ),
            minimum_keyword_matches=2,
            requires_retrieval=True,
        ),
        IntentRule(
            "iam_review",
            Intent.IAM_REVIEW,
            820,
            phrases=(
                "audit iam",
                "check least privilege",
                "review iam",
                "review this policy",
            ),
            keywords=("iam", "least privilege", "policy", "review"),
            minimum_keyword_matches=2,
            requires_retrieval=True,
        ),
        IntentRule(
            "pyspark_generation",
            Intent.PYSPARK_GENERATION,
            760,
            phrases=(
                "create pyspark",
                "generate pyspark",
                "write a pyspark",
                "write pyspark",
            ),
            keywords=(
                "build",
                "code",
                "create",
                "generate",
                "pyspark",
                "transformation",
                "write",
            ),
            minimum_keyword_matches=2,
        ),
        IntentRule(
            "sql_generation",
            Intent.SQL_GENERATION,
            750,
            phrases=(
                "create sql",
                "generate sql",
                "write a sql",
                "write sql",
            ),
            keywords=(
                "create",
                "generate",
                "query",
                "sql",
                "statement",
                "write",
            ),
            minimum_keyword_matches=2,
        ),
        IntentRule(
            "cdk_generation",
            Intent.CDK_GENERATION,
            740,
            phrases=(
                "create cdk",
                "generate cdk",
                "write a cdk",
                "write cdk",
            ),
            keywords=(
                "cdk",
                "construct",
                "create",
                "generate",
                "python",
                "write",
            ),
            minimum_keyword_matches=2,
        ),
        IntentRule(
            "pipeline_generation",
            Intent.PIPELINE_GENERATION,
            720,
            phrases=(
                "build a pipeline",
                "build pipeline",
                "create a pipeline",
                "create pipeline",
                "generate a pipeline",
            ),
            keywords=(
                "build",
                "create",
                "ingest",
                "pipeline",
                "s3",
                "source",
            ),
            minimum_keyword_matches=2,
        ),
        IntentRule(
            "pipeline_requirements",
            Intent.PIPELINE_REQUIREMENTS,
            700,
            phrases=(
                "define pipeline requirements",
                "gather pipeline requirements",
                "pipeline requirements",
                "requirements for a pipeline",
            ),
            keywords=(
                "constraints",
                "pipeline",
                "requirements",
                "sla",
                "source",
                "target",
            ),
            minimum_keyword_matches=2,
        ),
        IntentRule(
            "data_quality",
            Intent.DATA_QUALITY,
            620,
            phrases=(
                "data quality",
                "quality checks",
                "validate data",
            ),
            keywords=(
                "completeness",
                "data",
                "duplicates",
                "quality",
                "validation",
            ),
            minimum_keyword_matches=2,
            requires_retrieval=True,
        ),
        IntentRule(
            "monitoring",
            Intent.MONITORING_REQUEST,
            610,
            phrases=(
                "check alarm status",
                "check pipeline status",
                "monitor pipeline",
                "monitoring request",
                "show current alarms",
            ),
            keywords=(
                "alarm",
                "check",
                "cloudwatch",
                "current",
                "monitor",
                "monitoring",
                "status",
            ),
            minimum_keyword_matches=2,
            requires_tool_call=True,
        ),
        IntentRule(
            "cost",
            Intent.COST_QUESTION,
            600,
            phrases=(
                "aws cost",
                "cost estimate",
                "how much does",
                "how much is",
                "how much will",
                "reduce costs",
            ),
            keywords=(
                "aws",
                "budget",
                "cost",
                "expensive",
                "price",
                "spend",
            ),
            minimum_keyword_matches=2,
            requires_retrieval=True,
        ),
        IntentRule(
            "architecture",
            Intent.ARCHITECTURE_DESIGN,
            580,
            phrases=(
                "architecture design",
                "design an architecture",
                "design architecture",
                "solution architecture",
            ),
            keywords=(
                "architecture",
                "design",
                "event driven",
                "scalable",
                "serverless",
            ),
            minimum_keyword_matches=2,
            requires_retrieval=True,
        ),
        IntentRule(
            "knowledge_question",
            Intent.KNOWLEDGE_QUESTION,
            200,
            phrases=(
                "according to the documentation",
                "find in the runbook",
                "what does the documentation say",
                "what is in the knowledge base",
            ),
            keywords=(
                "documentation",
                "knowledge base",
                "runbook",
                "search",
            ),
            minimum_keyword_matches=1,
            requires_retrieval=True,
        ),
        IntentRule(
            "general_conversation",
            Intent.GENERAL_CONVERSATION,
            100,
            phrases=(
                "good afternoon",
                "good morning",
                "hello",
                "hi there",
                "how are you",
                "thank you",
            ),
            keywords=("hello", "thanks"),
            minimum_keyword_matches=1,
            preferred_knowledge_scope=KnowledgeScope.NONE,
        ),
    )


_REASONING_SUMMARIES = {
    Intent.ARCHITECTURE_DESIGN: "The request asks for architecture guidance.",
    Intent.PIPELINE_REQUIREMENTS: (
        "The request focuses on gathering pipeline requirements."
    ),
    Intent.PIPELINE_GENERATION: "The request asks to build a data pipeline.",
    Intent.PIPELINE_TROUBLESHOOTING: (
        "The request describes a pipeline problem to diagnose."
    ),
    Intent.AWS_ERROR_EXPLANATION: (
        "The request asks for an AWS error explanation."
    ),
    Intent.SQL_GENERATION: "The request asks for SQL code.",
    Intent.PYSPARK_GENERATION: "The request asks for PySpark code.",
    Intent.CDK_GENERATION: "The request asks for AWS CDK code.",
    Intent.IAM_REVIEW: "The request asks for an IAM or policy review.",
    Intent.DATA_QUALITY: "The request concerns data quality.",
    Intent.MONITORING_REQUEST: (
        "The request asks for current monitoring information."
    ),
    Intent.COST_QUESTION: "The request asks about AWS cost.",
    Intent.KNOWLEDGE_QUESTION: (
        "The request should be answered from scoped knowledge."
    ),
    Intent.DEPLOYMENT_REQUEST: (
        "The request asks to perform a deployment action."
    ),
    Intent.DESTRUCTIVE_ACTION_REQUEST: (
        "The request may modify or remove resources or data."
    ),
    Intent.GENERAL_CONVERSATION: (
        "The request is conversational and needs no specialist workflow."
    ),
    Intent.UNKNOWN: (
        "The request does not match a supported intent with enough confidence."
    ),
}


class RuleBasedIntentClassifier:
    """Network-free classifier with deterministic, prioritized matching."""

    def __init__(
        self,
        config: ClassificationRoutingConfig | None = None,
        *,
        rules: Sequence[IntentRule] | None = None,
    ) -> None:
        self._config = config or ClassificationRoutingConfig()
        self._rules = tuple(rules) if rules is not None else _default_rules()
        names = [rule.name for rule in self._rules]
        if len(names) != len(set(names)):
            raise ValueError("Rule names must be unique")

    @property
    def classifier_version(self) -> str:
        return self._config.classifier_version

    def classify(
        self,
        query: str,
        *,
        conversation_context: str | None = None,
        client_id: str | None = None,
        environment: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ClassificationResult:
        del client_id, environment, metadata
        normalized_query = normalize_request_text(query)
        normalized_context = normalize_request_text(
            conversation_context or ""
        )
        combined = " ".join(
            value for value in (normalized_query, normalized_context) if value
        )
        if not normalized_query:
            return self._unknown()

        is_discussion = normalized_query.startswith(_DISCUSSION_PREFIXES)
        if not is_discussion and any(
            re.search(pattern, combined)
            for pattern in _DESTRUCTIVE_PATTERNS
        ):
            return self._safety_result(
                Intent.DESTRUCTIVE_ACTION_REQUEST,
                "safety.destructive_action",
            )
        if not is_discussion and any(
            re.search(pattern, combined)
            for pattern in _DEPLOYMENT_PATTERNS
        ):
            return self._safety_result(
                Intent.DEPLOYMENT_REQUEST,
                "safety.deployment_action",
            )

        matches = [
            match
            for rule in self._rules
            if (match := self._match_rule(rule, combined)) is not None
        ]
        if not matches:
            if is_discussion:
                return ClassificationResult(
                    intent=Intent.KNOWLEDGE_QUESTION,
                    confidence=max(
                        self._config.minimum_confidence,
                        0.65,
                    ),
                    requires_retrieval=True,
                    requires_tool_call=False,
                    requires_approval=False,
                    preferred_knowledge_scope=(
                        KnowledgeScope.CLIENT_ENVIRONMENT
                    ),
                    reasoning_summary=_REASONING_SUMMARIES[
                        Intent.KNOWLEDGE_QUESTION
                    ],
                    matched_rules=("context.discussion",),
                )
            return self._unknown()

        matches.sort(
            key=lambda match: (
                match.confidence,
                match.rule.priority,
                match.rule.name,
            ),
            reverse=True,
        )
        selected = matches[0]
        confidence = selected.confidence
        ambiguous_names = [selected.rule.name]
        if len(matches) > 1:
            runner_up = matches[1]
            if (
                abs(
                    selected.rule.priority - runner_up.rule.priority
                )
                <= 20
                and abs(
                    selected.confidence - runner_up.confidence
                )
                <= 0.12
            ):
                confidence = max(
                    self._config.minimum_confidence,
                    confidence - 0.10,
                )
                ambiguous_names.append(runner_up.rule.name)

        if confidence < max(
            self._config.minimum_confidence,
            self._config.unknown_threshold,
        ):
            return self._unknown(tuple(ambiguous_names))

        intent = selected.rule.intent
        return ClassificationResult(
            intent=intent,
            confidence=round(confidence, 3),
            requires_retrieval=selected.rule.requires_retrieval,
            requires_tool_call=selected.rule.requires_tool_call,
            requires_approval=intent in self._config.approval_required_intents,
            preferred_knowledge_scope=(
                selected.rule.preferred_knowledge_scope
            ),
            reasoning_summary=_REASONING_SUMMARIES[intent],
            matched_rules=tuple(ambiguous_names),
        )

    @staticmethod
    def _match_rule(
        rule: IntentRule,
        normalized_text: str,
    ) -> _RuleMatch | None:
        normalized_phrases = tuple(
            normalize_request_text(value) for value in rule.phrases
        )
        normalized_keywords = tuple(
            normalize_request_text(value) for value in rule.keywords
        )
        phrase_matches = sum(
            _contains_term(normalized_text, phrase)
            for phrase in normalized_phrases
        )
        keyword_matches = sum(
            _contains_term(normalized_text, keyword)
            for keyword in normalized_keywords
        )
        if (
            phrase_matches == 0
            and keyword_matches < rule.minimum_keyword_matches
        ):
            return None
        confidence = min(
            0.99,
            0.58
            + min(phrase_matches, 2) * 0.14
            + min(keyword_matches, 4) * 0.045
            + min(rule.priority, 1_000) / 10_000,
        )
        return _RuleMatch(rule=rule, confidence=confidence)

    def _safety_result(
        self,
        intent: Intent,
        matched_rule: str,
    ) -> ClassificationResult:
        return ClassificationResult(
            intent=intent,
            confidence=0.99,
            requires_retrieval=False,
            requires_tool_call=True,
            requires_approval=True,
            preferred_knowledge_scope=KnowledgeScope.CLIENT_ENVIRONMENT,
            reasoning_summary=_REASONING_SUMMARIES[intent],
            matched_rules=(matched_rule,),
        )

    def _unknown(
        self,
        matched_rules: tuple[str, ...] = (),
    ) -> ClassificationResult:
        return ClassificationResult(
            intent=Intent.UNKNOWN,
            confidence=0.0,
            requires_retrieval=False,
            requires_tool_call=False,
            requires_approval=False,
            preferred_knowledge_scope=KnowledgeScope.NONE,
            reasoning_summary=_REASONING_SUMMARIES[Intent.UNKNOWN],
            matched_rules=matched_rules,
        )


def _contains_term(normalized_text: str, normalized_term: str) -> bool:
    return (
        re.search(
            rf"(?<!\w){re.escape(normalized_term)}(?!\w)",
            normalized_text,
        )
        is not None
    )


class BedrockIntentClassifier:
    """Future Bedrock classifier placeholder; it performs no AWS call."""

    def __init__(
        self,
        *,
        model_id: str,
        classifier_version: str,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id cannot be empty")
        if not classifier_version.strip():
            raise ValueError("classifier_version cannot be empty")
        self.model_id = model_id.strip()
        self._classifier_version = classifier_version.strip()

    @property
    def classifier_version(self) -> str:
        return self._classifier_version

    def classify(
        self,
        query: str,
        *,
        conversation_context: str | None = None,
        client_id: str | None = None,
        environment: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ClassificationResult:
        del query, conversation_context, client_id, environment, metadata
        # TODO: Add schema-constrained structured JSON responses.
        # TODO: Add prompt versioning and confidence calibration.
        # TODO: Fall back to deterministic rules on provider failure.
        # TODO: Add labeled Bedrock classification evaluation.
        raise NotImplementedError(
            "Bedrock intent classification is intentionally deferred"
        )
