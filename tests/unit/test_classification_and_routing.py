import json
import logging
from typing import Any, Mapping

import pytest

from knowledge.classification import (
    BedrockIntentClassifier,
    IntentRule,
    RuleBasedIntentClassifier,
    normalize_request_text,
)
from knowledge.classification_evaluation import (
    ClassificationEvaluationCase,
    ClassificationEvaluator,
    REPRESENTATIVE_CLASSIFICATION_CASES,
)
from knowledge.config import ClassificationRoutingConfig
from knowledge.intents import (
    ClassificationResult,
    Intent,
    IntentClassifier,
    KnowledgeScope,
)
from knowledge.routing import (
    RequestRouter,
    Route,
    ToolCategory,
)


@pytest.mark.parametrize(
    ("query", "expected_intent"),
    [
        (
            "Design a scalable serverless architecture",
            Intent.ARCHITECTURE_DESIGN,
        ),
        (
            "Gather pipeline requirements and source constraints",
            Intent.PIPELINE_REQUIREMENTS,
        ),
        (
            "Build PostgreSQL to S3 pipeline",
            Intent.PIPELINE_GENERATION,
        ),
        (
            "My Glue job fail; troubleshoot the pipeline",
            Intent.PIPELINE_TROUBLESHOOTING,
        ),
        (
            "Explain this AWS access denied error",
            Intent.AWS_ERROR_EXPLANATION,
        ),
        ("Write SQL query for daily totals", Intent.SQL_GENERATION),
        (
            "Write PySpark transformation code",
            Intent.PYSPARK_GENERATION,
        ),
        (
            "Generate CDK Python construct",
            Intent.CDK_GENERATION,
        ),
        ("Review IAM policy", Intent.IAM_REVIEW),
        ("Add data quality checks", Intent.DATA_QUALITY),
        ("Check current alarm status", Intent.MONITORING_REQUEST),
        ("Estimate AWS cost", Intent.COST_QUESTION),
        ("Find this in the runbook", Intent.KNOWLEDGE_QUESTION),
        ("Deploy CDK", Intent.DEPLOYMENT_REQUEST),
        (
            "Delete the production bucket",
            Intent.DESTRUCTIVE_ACTION_REQUEST,
        ),
        ("Hello", Intent.GENERAL_CONVERSATION),
        ("florbulate this", Intent.UNKNOWN),
    ],
)
def test_rule_classifier_covers_every_intent(query, expected_intent):
    result = RuleBasedIntentClassifier().classify(query)

    assert result.intent == expected_intent
    assert 0.0 <= result.confidence <= 1.0
    assert result.reasoning_summary


def test_representative_dataset_covers_all_intents_and_scores_perfectly():
    expected = {case.expected_intent for case in REPRESENTATIVE_CLASSIFICATION_CASES}

    summary = ClassificationEvaluator().evaluate(
        RuleBasedIntentClassifier(),
        REPRESENTATIVE_CLASSIFICATION_CASES,
    )

    assert expected == set(Intent)
    assert summary.accuracy == 1.0
    assert summary.unknown_rate == pytest.approx(1 / len(Intent))
    assert all(
        metrics.precision == 1.0
        and metrics.recall == 1.0
        and metrics.f1 == 1.0
        for metrics in summary.per_intent.values()
    )


def test_classifier_is_deterministic_and_implements_separate_protocol():
    classifier = RuleBasedIntentClassifier()

    first = classifier.classify(
        "WRITE a PySpark_transformation!",
        conversation_context="Use our normal patterns.",
        client_id="client-a",
        environment="dev",
        metadata={"channel": "test"},
    )
    second = classifier.classify(
        "WRITE a PySpark_transformation!",
        conversation_context="Use our normal patterns.",
        client_id="client-a",
        environment="dev",
        metadata={"channel": "test"},
    )

    assert isinstance(classifier, IntentClassifier)
    assert first == second
    assert normalize_request_text("PySpark_transform—Now") == (
        "pyspark transform now"
    )


def test_ambiguous_rules_use_priority_and_reduce_confidence():
    rules = (
        IntentRule(
            "higher",
            Intent.SQL_GENERATION,
            100,
            phrases=("build query",),
        ),
        IntentRule(
            "runner_up",
            Intent.PIPELINE_GENERATION,
            90,
            phrases=("build query",),
        ),
    )

    result = RuleBasedIntentClassifier(rules=rules).classify(
        "Build query"
    )

    assert result.intent == Intent.SQL_GENERATION
    assert result.confidence < 0.8
    assert result.matched_rules == ("higher", "runner_up")


def test_custom_rules_are_configurable():
    custom = IntentRule(
        "custom_phrase",
        Intent.DATA_QUALITY,
        10,
        phrases=("quarantine invalid rows",),
        preferred_knowledge_scope=KnowledgeScope.CLIENT,
    )

    result = RuleBasedIntentClassifier(rules=(custom,)).classify(
        "Please quarantine invalid rows"
    )

    assert result.intent == Intent.DATA_QUALITY
    assert result.preferred_knowledge_scope == KnowledgeScope.CLIENT
    assert result.matched_rules == ("custom_phrase",)


def test_conversation_context_distinguishes_iam_error_from_review():
    classifier = RuleBasedIntentClassifier()

    error = classifier.classify(
        "What does this mean?",
        conversation_context="AWS IAM access denied error",
    )
    review = classifier.classify(
        "Review this policy for least privilege",
        conversation_context="IAM role for the ingestion job",
    )

    assert error.intent == Intent.AWS_ERROR_EXPLANATION
    assert review.intent == Intent.IAM_REVIEW


@pytest.mark.parametrize(
    "query",
    [
        "Deploy the CDK stack",
        "Apply the CloudFormation stack",
        "Release the application to production",
    ],
)
def test_deployment_actions_always_require_approval(query):
    result = RuleBasedIntentClassifier().classify(query)
    plan = RequestRouter().route(result)

    assert result.intent == Intent.DEPLOYMENT_REQUEST
    assert result.requires_approval is True
    assert plan.selected_route == Route.APPROVAL_REQUIRED
    assert plan.approval_required is True
    assert plan.tool_category == ToolCategory.DEPLOYMENT


@pytest.mark.parametrize(
    "query",
    [
        "Delete the prod bucket",
        "Destroy this stack",
        "Destroy everything",
        "Delete this resource",
        "Overwrite production data",
        "Replace the existing resources",
        "Move customer data",
        "Modify production resources",
        "Broaden IAM permissions",
    ],
)
def test_destructive_actions_require_approval_and_safety_review(query):
    result = RuleBasedIntentClassifier().classify(query)
    plan = RequestRouter().route(result)

    assert result.intent == Intent.DESTRUCTIVE_ACTION_REQUEST
    assert result.requires_approval is True
    assert plan.selected_route == Route.REJECTION_OR_SAFETY_REVIEW
    assert plan.approval_required is True
    assert plan.safety_review_required is True


@pytest.mark.parametrize(
    "query",
    [
        "Explain why deleting a production bucket is dangerous",
        "How would a CDK deployment work?",
        "Please explain how to deploy the CDK stack",
        "Should we delete the production bucket?",
        "What happens if we overwrite data?",
    ],
)
def test_discussion_of_sensitive_actions_is_not_execution_permission(query):
    result = RuleBasedIntentClassifier().classify(query)
    plan = RequestRouter().route(result)

    assert result.intent not in {
        Intent.DEPLOYMENT_REQUEST,
        Intent.DESTRUCTIVE_ACTION_REQUEST,
    }
    assert plan.approval_required is False
    assert plan.selected_route == Route.RETRIEVAL


def test_router_safety_overrides_cannot_be_disabled_by_provider_result():
    unsafe_provider_result = ClassificationResult(
        intent=Intent.DESTRUCTIVE_ACTION_REQUEST,
        confidence=0.9,
        requires_retrieval=False,
        requires_tool_call=False,
        requires_approval=False,
        preferred_knowledge_scope=KnowledgeScope.NONE,
        reasoning_summary="A destructive action was requested.",
    )

    plan = RequestRouter().route(unsafe_provider_result)

    assert plan.approval_required is True
    assert plan.safety_review_required is True
    assert plan.selected_route == Route.REJECTION_OR_SAFETY_REVIEW


@pytest.mark.parametrize(
    ("query", "route"),
    [
        ("Hello", Route.DIRECT_RESPONSE),
        ("Find this in the runbook", Route.RETRIEVAL),
        (
            "Define pipeline requirements",
            Route.REQUIREMENTS_GATHERING,
        ),
        ("Write SQL query", Route.CODE_GENERATION),
        (
            "My pipeline failed",
            Route.TROUBLESHOOTING,
        ),
        ("Check current alarm status", Route.TOOL_EXECUTION),
        ("Deploy CDK", Route.APPROVAL_REQUIRED),
        (
            "Delete the production bucket",
            Route.REJECTION_OR_SAFETY_REVIEW,
        ),
    ],
)
def test_router_exposes_every_route_without_executing_tools(query, route):
    classifier = RuleBasedIntentClassifier()

    plan = RequestRouter().route(classifier.classify(query))

    assert plan.selected_route == route


def test_routing_preserves_client_environment_and_retrieval_settings():
    config = ClassificationRoutingConfig(default_retrieval_top_k=9)
    result = RuleBasedIntentClassifier(config).classify(
        "Find the incident process in the runbook",
        client_id="client-a",
        environment="stage",
    )

    client_a = RequestRouter(config).route(
        result,
        client_id="client-a",
        environment="stage",
    )
    client_b = RequestRouter(config).route(
        result,
        client_id="client-b",
        environment="prod",
    )

    assert client_a.retrieval_top_k == 9
    assert client_a.retrieval_scope == KnowledgeScope.CLIENT_ENVIRONMENT
    assert (client_a.client_id, client_a.environment) == (
        "client-a",
        "stage",
    )
    assert (client_b.client_id, client_b.environment) == (
        "client-b",
        "prod",
    )
    assert client_a != client_b


def test_routing_logs_safe_structured_fields_only(caplog):
    logger = logging.getLogger("test.request.routing")
    caplog.set_level(logging.INFO, logger=logger.name)
    secret_query = "Find credential-value-123 in the runbook"
    classifier = RuleBasedIntentClassifier()

    RequestRouter(event_logger=logger).route(
        classifier.classify(secret_query),
        client_id="client-a",
        environment="dev",
    )

    event = json.loads(caplog.records[0].message)
    assert event["classifier_version"] == "rules-v1"
    assert event["predicted_intent"] == "knowledge_question"
    assert event["route"] == "retrieval"
    assert event["client_id"] == "client-a"
    assert event["environment"] == "dev"
    assert event["elapsed_ms"] >= 0
    assert secret_query not in caplog.text
    assert "credential-value-123" not in caplog.text
    assert "reasoning_summary" not in caplog.text


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"minimum_confidence": -0.1}, "minimum_confidence"),
        ({"unknown_threshold": 1.1}, "unknown_threshold"),
        (
            {"minimum_confidence": 0.4, "unknown_threshold": 0.5},
            "unknown_threshold",
        ),
        ({"default_retrieval_top_k": 0}, "default_retrieval_top_k"),
        ({"classifier_version": " "}, "classifier_version"),
        (
            {
                "approval_required_intents": frozenset(
                    {Intent.DESTRUCTIVE_ACTION_REQUEST}
                )
            },
            "Deployment and destructive",
        ),
        (
            {"safety_review_intents": frozenset()},
            "Destructive actions",
        ),
    ],
)
def test_classification_config_rejects_unsafe_or_invalid_values(
    kwargs,
    message,
):
    with pytest.raises(ValueError, match=message):
        ClassificationRoutingConfig(**kwargs)


def test_evaluator_reports_confusion_matrix_and_unknown_rate():
    class MappingClassifier:
        classifier_version = "mapping-v1"

        def classify(
            self,
            query: str,
            *,
            conversation_context: str | None = None,
            client_id: str | None = None,
            environment: str | None = None,
            metadata: Mapping[str, Any] | None = None,
        ) -> ClassificationResult:
            del conversation_context, client_id, environment, metadata
            intent = (
                Intent.SQL_GENERATION
                if query == "correct"
                else Intent.UNKNOWN
            )
            return ClassificationResult(
                intent=intent,
                confidence=0.8,
                requires_retrieval=False,
                requires_tool_call=False,
                requires_approval=False,
                preferred_knowledge_scope=KnowledgeScope.NONE,
                reasoning_summary="A fixed test prediction.",
            )

    summary = ClassificationEvaluator().evaluate(
        MappingClassifier(),
        (
            ClassificationEvaluationCase(
                "correct",
                Intent.SQL_GENERATION,
            ),
            ClassificationEvaluationCase(
                "missed",
                Intent.PYSPARK_GENERATION,
            ),
        ),
    )

    assert summary.accuracy == 0.5
    assert summary.unknown_rate == 0.5
    assert summary.confusion_matrix[Intent.SQL_GENERATION][
        Intent.SQL_GENERATION
    ] == 1
    assert summary.confusion_matrix[Intent.PYSPARK_GENERATION][
        Intent.UNKNOWN
    ] == 1
    assert summary.per_intent[Intent.SQL_GENERATION].f1 == 1.0
    assert summary.per_intent[Intent.PYSPARK_GENERATION].recall == 0.0


def test_bedrock_classifier_is_placeholder_and_makes_no_call():
    classifier = BedrockIntentClassifier(
        model_id="future-model",
        classifier_version="bedrock-v0",
    )

    assert isinstance(classifier, IntentClassifier)
    with pytest.raises(NotImplementedError, match="deferred"):
        classifier.classify("Classify this")
