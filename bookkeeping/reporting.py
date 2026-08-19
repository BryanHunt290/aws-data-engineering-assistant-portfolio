"""Deterministic bookkeeping reports with optional advisory explanation."""

from decimal import Decimal
import json
import re

from bookkeeping.analytics import BookkeepingAnalyticsService
from bookkeeping.duplicates import DuplicateDetector
from bookkeeping.models import (
    BookkeepingAnalytics,
    BookkeepingTransaction,
    BusinessReport,
    DuplicateCandidate,
)
from bookkeeping.grounded_answers import GroundedBookkeepingAnswerService
from bookkeeping.knowledge_models import (
    BookkeepingCitation,
    BookkeepingRetrievalMode,
    GroundedBookkeepingAnswer,
)
from bookkeeping.knowledge_service import format_bookkeeping_citation
from knowledge.llm import LLMProvider
from knowledge.llm_errors import LLMProviderError


_REPORT_SYSTEM_PROMPT = (
    "Explain the supplied deterministic bookkeeping metrics for a business "
    "owner. Do not recalculate, replace, or contradict totals. Do not provide "
    "accounting or tax advice and do not claim tax deductibility. Identify "
    "review questions, not conclusions. Return concise Markdown."
)


class BusinessReportService:
    """Combine immutable calculations with optional LLM commentary."""

    def __init__(
        self,
        *,
        analytics_service: BookkeepingAnalyticsService | None = None,
        duplicate_detector: DuplicateDetector | None = None,
        llm_provider: LLMProvider | None = None,
        grounded_answer_service: GroundedBookkeepingAnswerService | None = None,
    ) -> None:
        self._analytics = (
            analytics_service or BookkeepingAnalyticsService()
        )
        self._duplicates = duplicate_detector or DuplicateDetector()
        self._llm_provider = llm_provider
        self._grounded_answer_service = grounded_answer_service

    def generate(
        self,
        transactions: tuple[BookkeepingTransaction, ...],
        *,
        include_ai_explanation: bool = False,
        include_reference_context: bool = False,
        knowledge_question: str = (
            "Explain this bookkeeping summary using approved procedures, "
            "categorization policy, and duplicate-review guidance."
        ),
        client_id: str | None = None,
        retrieval_mode: BookkeepingRetrievalMode | str = (
            BookkeepingRetrievalMode.KEYWORD
        ),
        category_suggestions=(),
    ) -> BusinessReport:
        analytics = self._analytics.analyze(transactions)
        duplicates = self._duplicates.detect(transactions)
        explanation = None
        provider_name = None
        warnings: list[str] = []
        grounded_answer: GroundedBookkeepingAnswer | None = None
        reference_guidance: tuple[str, ...] = ()
        citations: tuple[BookkeepingCitation, ...] = ()
        human_review_items: list[str] = []
        if include_ai_explanation:
            if include_reference_context:
                if self._grounded_answer_service is None:
                    warnings.append(
                        "Reference-grounded explanation was requested but no "
                        "grounded answer service is configured."
                    )
                else:
                    try:
                        grounded_answer = self._grounded_answer_service.answer(
                            knowledge_question,
                            analytics=analytics,
                            duplicate_candidates=duplicates,
                            category_suggestions=category_suggestions,
                            client_id=client_id,
                            retrieval_mode=retrieval_mode,
                        )
                        explanation = grounded_answer.answer_text
                        provider_name = grounded_answer.provider_name
                        citations = grounded_answer.retrieved_citations
                        reference_guidance = tuple(
                            _guidance_preview(passage.text)
                            for passage in grounded_answer.retrieved_passages
                        )
                        warnings.extend(grounded_answer.warnings)
                        if grounded_answer.sources_conflict:
                            human_review_items.append(
                                "Retrieved references conflict; resolve the "
                                "policy conflict before acting."
                            )
                    except LLMProviderError:
                        warnings.append(
                            "Reference-grounded AI explanation was unavailable; "
                            "deterministic reporting remains complete."
                        )
            elif self._llm_provider is None:
                warnings.append(
                    "AI explanation was requested but no provider is configured."
                )
            else:
                provider_name = self._llm_provider.provider_name
                try:
                    result = self._llm_provider.generate(
                        system_prompt=_REPORT_SYSTEM_PROMPT,
                        user_prompt=self._report_prompt(
                            analytics,
                            duplicates,
                        ),
                        model_parameters={
                            "temperature": 0,
                            "maximum_tokens": 1_024,
                        },
                    )
                    explanation = result.generated_text.strip()
                    if not explanation:
                        raise ValueError("AI explanation was empty")
                    if re.search(
                        r"\b(?:tax\w*|deductib\w*)\b",
                        explanation,
                        flags=re.IGNORECASE,
                    ):
                        raise ValueError(
                            "AI explanation contained prohibited advice"
                        )
                except (LLMProviderError, ValueError):
                    warnings.append(
                        "AI explanation was unavailable; deterministic "
                        "reporting remains complete."
                    )
                    explanation = None

        if duplicates:
            human_review_items.append(
                f"Review {len(duplicates)} likely duplicate candidate(s); none "
                "were deleted or merged."
            )
        if analytics.uncategorized_transaction_count:
            human_review_items.append(
                "Review uncategorized transactions before approving categories."
            )
        if category_suggestions:
            human_review_items.append(
                "Every category suggestion is advisory and remains unapproved."
            )
        if not human_review_items:
            human_review_items.append(
                "Review the report with a qualified human before relying on it."
            )

        markdown = self._markdown(
            analytics,
            duplicates,
            explanation,
            provider_name,
            tuple(warnings),
            reference_guidance,
            citations,
            tuple(human_review_items),
        )
        return BusinessReport(
            analytics=analytics,
            duplicate_candidates=duplicates,
            ai_explanation=explanation,
            explanation_provider=(
                provider_name if explanation is not None else None
            ),
            warnings=tuple(warnings),
            markdown=markdown,
            grounded_answer=grounded_answer,
            reference_guidance=reference_guidance,
            citations=citations,
            human_review_items=tuple(human_review_items),
        )

    @staticmethod
    def _report_prompt(
        analytics: BookkeepingAnalytics,
        duplicates: tuple[DuplicateCandidate, ...],
    ) -> str:
        payload = {
            "deterministic_metrics": {
                "total_income": str(analytics.total_income),
                "total_expenses": str(analytics.total_expenses),
                "net_cash_flow": str(analytics.net_cash_flow),
                "transaction_count": analytics.transaction_count,
                "uncategorized_transaction_count": (
                    analytics.uncategorized_transaction_count
                ),
                "uncategorized_expense_percentage": str(
                    analytics.uncategorized_expense_percentage
                ),
                "monthly": [
                    {
                        "period": item.period,
                        "income": str(item.income),
                        "expenses": str(item.expenses),
                        "net_cash_flow": str(item.net_cash_flow),
                    }
                    for item in analytics.monthly
                ],
                "category_totals": [
                    {
                        "category": item.name,
                        "expenses": str(item.expenses),
                        "net_cash_flow": str(item.net_cash_flow),
                    }
                    for item in analytics.totals_by_category
                ],
                "largest_expenses": [
                    {
                        "reference": item.transaction_reference,
                        "date": item.transaction_date.isoformat(),
                        "amount": str(item.amount),
                        "category": item.category or "Uncategorized",
                    }
                    for item in analytics.largest_expenses
                ],
                "duplicate_candidates": [
                    {
                        "first_reference": item.first_reference,
                        "second_reference": item.second_reference,
                        "rule": item.rule.value,
                        "confidence": str(item.confidence),
                    }
                    for item in duplicates
                ],
            },
            "instructions": (
                "Explain notable changes, spending concentrations, duplicate "
                "review candidates, and uncategorized risk. Ask review "
                "questions. Do not restate private descriptions or accounts."
            ),
        }
        return json.dumps(payload, sort_keys=True)

    @staticmethod
    def _markdown(
        analytics: BookkeepingAnalytics,
        duplicates: tuple[DuplicateCandidate, ...],
        explanation: str | None,
        provider_name: str | None,
        warnings: tuple[str, ...],
        reference_guidance: tuple[str, ...] = (),
        citations: tuple[BookkeepingCitation, ...] = (),
        human_review_items: tuple[str, ...] = (),
    ) -> str:
        lines = [
            "# Bookkeeping business summary",
            "",
            "> Local analytical report. This is not accounting or tax advice.",
            "",
            "## A. Deterministic calculations",
            "",
            f"- Total income: {_money(analytics.total_income)}",
            f"- Total expenses: {_money(analytics.total_expenses)}",
            f"- Net cash flow: {_money(analytics.net_cash_flow)}",
            f"- Transaction count: {analytics.transaction_count}",
            (
                "- Average signed transaction amount: "
                f"{_money(analytics.average_transaction_amount)}"
            ),
            (
                "- Uncategorized transactions: "
                f"{analytics.uncategorized_transaction_count}"
            ),
            (
                "- Uncategorized share of expense dollars: "
                f"{analytics.uncategorized_expense_percentage:.2f}%"
            ),
            f"- Likely duplicate candidates: {len(duplicates)}",
            "",
            "## Monthly summary",
            "",
            "| Month | Income | Expenses | Net cash flow |",
            "| --- | ---: | ---: | ---: |",
        ]
        lines.extend(
            (
                f"| {item.period} | {_money(item.income)} | "
                f"{_money(item.expenses)} | "
                f"{_money(item.net_cash_flow)} |"
            )
            for item in analytics.monthly
        )
        lines.extend(["", "## B. AI-generated explanation", ""])
        if explanation:
            lines.extend(
                [
                    (
                        f"*Provider: {provider_name}. Advisory output requiring "
                        "human review.*"
                    ),
                    "",
                    explanation,
                ]
            )
        else:
            lines.append(
                "No AI explanation was generated. Deterministic calculations "
                "above remain complete."
            )
        lines.extend(["", "## C. Reference guidance", ""])
        if reference_guidance:
            lines.extend(
                f"- [{index}] {guidance}"
                for index, guidance in enumerate(
                    reference_guidance,
                    start=1,
                )
            )
        else:
            lines.append(
                "No approved reference guidance was included in this report."
            )
        lines.extend(["", "## D. Citations", ""])
        if citations:
            lines.extend(
                f"- {format_bookkeeping_citation(citation)}"
                for citation in citations
            )
        else:
            lines.append("No retrieved citations were used.")
        lines.extend(["", "## E. Items requiring human review", ""])
        lines.extend(f"- {item}" for item in human_review_items)
        if warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {warning}" for warning in warnings)
        return "\n".join(lines) + "\n"


def _money(value: Decimal) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def _guidance_preview(text: str, limit: int = 240) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"
