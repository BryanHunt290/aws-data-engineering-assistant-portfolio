"""Strongly typed models for normalized bookkeeping analysis."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum

from bookkeeping.knowledge_models import (
    BookkeepingCitation,
    GroundedBookkeepingAnswer,
)


@dataclass(frozen=True)
class BookkeepingTransaction:
    """One normalized immutable transaction.

    Positive amounts are inflows and negative amounts are outflows.
    """

    transaction_date: date
    description: str
    amount: Decimal
    source_row_number: int
    transaction_id: str | None = None
    account_name: str | None = None
    category: str | None = None
    memo: str | None = None
    vendor_or_payee: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.transaction_date, date):
            raise ValueError("transaction_date must be a date")
        if not isinstance(self.description, str):
            raise ValueError("description must be a string")
        description = self.description.strip()
        if not description:
            raise ValueError("description cannot be empty")
        if not isinstance(self.amount, Decimal) or not self.amount.is_finite():
            raise ValueError("amount must be a finite Decimal")
        if self.source_row_number < 2:
            raise ValueError("source_row_number must be at least 2")
        object.__setattr__(self, "description", description)
        for field_name in (
            "transaction_id",
            "account_name",
            "category",
            "memo",
            "vendor_or_payee",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string or None")
            normalized = value.strip() if value and value.strip() else None
            object.__setattr__(self, field_name, normalized)

    @property
    def reference(self) -> str:
        return f"row-{self.source_row_number}"


@dataclass(frozen=True)
class CSVRowError:
    row_number: int
    error_code: str
    message: str


@dataclass(frozen=True)
class CSVIngestionSummary:
    rows_received: int
    rows_accepted: int
    rows_rejected: int
    warnings: tuple[str, ...] = ()
    row_errors: tuple[CSVRowError, ...] = ()


@dataclass(frozen=True)
class CSVIngestionResult:
    transactions: tuple[BookkeepingTransaction, ...]
    summary: CSVIngestionSummary


@dataclass(frozen=True)
class PeriodSummary:
    period: str
    income: Decimal
    expenses: Decimal
    net_cash_flow: Decimal


@dataclass(frozen=True)
class DimensionSummary:
    name: str
    income: Decimal
    expenses: Decimal
    net_cash_flow: Decimal
    transaction_count: int


@dataclass(frozen=True)
class ExpenseSummary:
    transaction_reference: str
    transaction_date: date
    description: str
    amount: Decimal
    category: str | None
    account_name: str | None


@dataclass(frozen=True)
class BookkeepingAnalytics:
    total_income: Decimal
    total_expenses: Decimal
    net_cash_flow: Decimal
    transaction_count: int
    average_transaction_amount: Decimal
    monthly: tuple[PeriodSummary, ...]
    totals_by_category: tuple[DimensionSummary, ...]
    totals_by_account: tuple[DimensionSummary, ...]
    largest_expenses: tuple[ExpenseSummary, ...]
    uncategorized_transaction_count: int
    uncategorized_expense_percentage: Decimal


class DuplicateRule(StrEnum):
    IDENTICAL_TRANSACTION_ID = "identical_transaction_id"
    SAME_DESCRIPTION_AMOUNT_DATE = "same_description_amount_date"
    SAME_DESCRIPTION_AMOUNT_NEAR_DATE = (
        "same_description_amount_near_date"
    )


@dataclass(frozen=True)
class DuplicateCandidate:
    first_reference: str
    second_reference: str
    rule: DuplicateRule
    confidence: Decimal
    explanation: str


@dataclass(frozen=True)
class CategorySuggestion:
    transaction_reference: str
    suggested_category: str
    confidence: Decimal
    rationale: str
    source: str
    requires_review: bool = True
    supporting_citations: tuple[BookkeepingCitation, ...] = ()
    policy_context_found: bool = False
    policy_conflict: bool = False
    context_basis: str = (
        "No approved policy context was requested; human review is required."
    )


@dataclass(frozen=True)
class BusinessReport:
    analytics: BookkeepingAnalytics
    duplicate_candidates: tuple[DuplicateCandidate, ...]
    ai_explanation: str | None
    explanation_provider: str | None
    warnings: tuple[str, ...]
    markdown: str
    grounded_answer: GroundedBookkeepingAnswer | None = None
    reference_guidance: tuple[str, ...] = ()
    citations: tuple[BookkeepingCitation, ...] = ()
    human_review_items: tuple[str, ...] = ()
