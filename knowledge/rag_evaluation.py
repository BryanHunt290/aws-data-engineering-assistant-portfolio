"""Deterministic end-to-end RAG evaluation without an LLM judge."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from knowledge.application import RAGApplicationService
from knowledge.application_models import (
    ApplicationRequest,
    ApplicationStatus,
)
from knowledge.intents import Intent


@dataclass(frozen=True)
class RAGEvaluationCase:
    """One labeled application request and deterministic assertions."""

    query: str
    expected_intent: Intent
    expected_source_ids: frozenset[str] = frozenset()
    reference_answer: str = ""
    required_facts: tuple[str, ...] = ()
    forbidden_claims: tuple[str, ...] = ()
    expect_insufficient_context: bool = False
    client_id: str = "evaluation-client"
    environment: str = "test"


@dataclass(frozen=True)
class RAGCaseMetrics:
    """Checks for one evaluated response."""

    intent_correct: bool
    source_recall: float
    required_facts_present: bool
    forbidden_claims_absent: bool
    insufficient_context_correct: bool


@dataclass(frozen=True)
class RAGEvaluationSummary:
    """Mean metrics and individual deterministic checks."""

    intent_accuracy: float
    source_recall: float
    required_fact_rate: float
    forbidden_claim_avoidance_rate: float
    insufficient_context_accuracy: float
    cases: tuple[RAGCaseMetrics, ...]


class RAGEvaluator:
    """Evaluate an application using fake providers and labeled cases."""

    def evaluate(
        self,
        application: RAGApplicationService,
        cases: Sequence[RAGEvaluationCase],
    ) -> RAGEvaluationSummary:
        if not cases:
            raise ValueError("At least one RAG evaluation case is required")

        metrics: list[RAGCaseMetrics] = []
        for index, case in enumerate(cases):
            response = application.handle(
                ApplicationRequest(
                    request_id=f"evaluation-{index:04d}",
                    query=case.query,
                    client_id=case.client_id,
                    environment=case.environment,
                    timestamp=datetime(
                        2026,
                        7,
                        27,
                        tzinfo=timezone.utc,
                    ),
                )
            )
            actual_source_ids = {
                source.source_id for source in response.sources
            }
            source_recall = (
                len(actual_source_ids & case.expected_source_ids)
                / len(case.expected_source_ids)
                if case.expected_source_ids
                else 1.0
            )
            normalized_answer = response.answer.casefold()
            metrics.append(
                RAGCaseMetrics(
                    intent_correct=response.intent == case.expected_intent,
                    source_recall=source_recall,
                    required_facts_present=all(
                        fact.casefold() in normalized_answer
                        for fact in case.required_facts
                    ),
                    forbidden_claims_absent=all(
                        claim.casefold() not in normalized_answer
                        for claim in case.forbidden_claims
                    ),
                    insufficient_context_correct=(
                        (
                            response.status
                            == ApplicationStatus.INSUFFICIENT_CONTEXT
                        )
                        == case.expect_insufficient_context
                    ),
                )
            )

        count = len(metrics)
        return RAGEvaluationSummary(
            intent_accuracy=sum(
                metric.intent_correct for metric in metrics
            )
            / count,
            source_recall=sum(
                metric.source_recall for metric in metrics
            )
            / count,
            required_fact_rate=sum(
                metric.required_facts_present for metric in metrics
            )
            / count,
            forbidden_claim_avoidance_rate=sum(
                metric.forbidden_claims_absent for metric in metrics
            )
            / count,
            insufficient_context_accuracy=sum(
                metric.insufficient_context_correct for metric in metrics
            )
            / count,
            cases=tuple(metrics),
        )
