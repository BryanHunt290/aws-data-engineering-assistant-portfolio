"""Labeled evaluation metrics for provider-neutral intent classifiers."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from knowledge.intents import Intent, IntentClassifier


@dataclass(frozen=True)
class ClassificationEvaluationCase:
    """One synthetic or curated labeled classification request."""

    query: str
    expected_intent: Intent
    conversation_context: str | None = None
    client_id: str | None = None
    environment: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class IntentMetrics:
    """Precision, recall, and F1 for one intent."""

    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ClassificationEvaluationSummary:
    """Aggregate classification quality and confusion-matrix data."""

    accuracy: float
    per_intent: Mapping[Intent, IntentMetrics]
    confusion_matrix: Mapping[Intent, Mapping[Intent, int]]
    unknown_rate: float
    expected_intents: tuple[Intent, ...]
    predicted_intents: tuple[Intent, ...]


class ClassificationEvaluator:
    """Evaluate any classifier that implements the intent protocol."""

    def evaluate(
        self,
        classifier: IntentClassifier,
        cases: Sequence[ClassificationEvaluationCase],
    ) -> ClassificationEvaluationSummary:
        if not cases:
            raise ValueError("At least one classification case is required")

        expected = tuple(case.expected_intent for case in cases)
        predicted = tuple(
            classifier.classify(
                case.query,
                conversation_context=case.conversation_context,
                client_id=case.client_id,
                environment=case.environment,
                metadata=case.metadata,
            ).intent
            for case in cases
        )
        pairs = Counter(zip(expected, predicted))
        intent_set = set(expected) | set(predicted)
        per_intent: dict[Intent, IntentMetrics] = {}
        matrix: dict[Intent, dict[Intent, int]] = defaultdict(dict)

        for expected_intent, predicted_intent in pairs:
            matrix[expected_intent][predicted_intent] = pairs[
                (expected_intent, predicted_intent)
            ]

        for intent in sorted(intent_set, key=lambda value: value.value):
            true_positive = pairs[(intent, intent)]
            false_positive = sum(
                count
                for (actual, guess), count in pairs.items()
                if guess == intent and actual != intent
            )
            false_negative = sum(
                count
                for (actual, guess), count in pairs.items()
                if actual == intent and guess != intent
            )
            precision = (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else 0.0
            )
            recall = (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative
                else 0.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            per_intent[intent] = IntentMetrics(
                precision=precision,
                recall=recall,
                f1=f1,
                support=sum(item == intent for item in expected),
            )

        total = len(cases)
        return ClassificationEvaluationSummary(
            accuracy=sum(
                actual == guess
                for actual, guess in zip(expected, predicted)
            )
            / total,
            per_intent=per_intent,
            confusion_matrix={
                actual: dict(predictions)
                for actual, predictions in matrix.items()
            },
            unknown_rate=sum(
                intent == Intent.UNKNOWN for intent in predicted
            )
            / total,
            expected_intents=expected,
            predicted_intents=predicted,
        )


REPRESENTATIVE_CLASSIFICATION_CASES = (
    ClassificationEvaluationCase(
        "Design a serverless data architecture",
        Intent.ARCHITECTURE_DESIGN,
    ),
    ClassificationEvaluationCase(
        "Define pipeline requirements and SLA constraints",
        Intent.PIPELINE_REQUIREMENTS,
    ),
    ClassificationEvaluationCase(
        "Build a pipeline from PostgreSQL to S3",
        Intent.PIPELINE_GENERATION,
    ),
    ClassificationEvaluationCase(
        "My Glue job is failing, troubleshoot the pipeline",
        Intent.PIPELINE_TROUBLESHOOTING,
    ),
    ClassificationEvaluationCase(
        "Explain this AWS access denied error",
        Intent.AWS_ERROR_EXPLANATION,
    ),
    ClassificationEvaluationCase(
        "Write SQL query to aggregate daily totals",
        Intent.SQL_GENERATION,
    ),
    ClassificationEvaluationCase(
        "Write a PySpark transformation for these records",
        Intent.PYSPARK_GENERATION,
    ),
    ClassificationEvaluationCase(
        "Generate CDK Python constructs for an event bus",
        Intent.CDK_GENERATION,
    ),
    ClassificationEvaluationCase(
        "Review IAM policy for least privilege",
        Intent.IAM_REVIEW,
    ),
    ClassificationEvaluationCase(
        "Create data quality checks for duplicates",
        Intent.DATA_QUALITY,
    ),
    ClassificationEvaluationCase(
        "Show current alarms and CloudWatch status",
        Intent.MONITORING_REQUEST,
    ),
    ClassificationEvaluationCase(
        "What is the AWS cost estimate",
        Intent.COST_QUESTION,
    ),
    ClassificationEvaluationCase(
        "Find this topic in the runbook",
        Intent.KNOWLEDGE_QUESTION,
    ),
    ClassificationEvaluationCase(
        "Deploy the CDK stack",
        Intent.DEPLOYMENT_REQUEST,
    ),
    ClassificationEvaluationCase(
        "Delete the production bucket",
        Intent.DESTRUCTIVE_ACTION_REQUEST,
    ),
    ClassificationEvaluationCase(
        "Hello, how are you",
        Intent.GENERAL_CONVERSATION,
    ),
    ClassificationEvaluationCase(
        "florbulate the translucent widget",
        Intent.UNKNOWN,
    ),
)
