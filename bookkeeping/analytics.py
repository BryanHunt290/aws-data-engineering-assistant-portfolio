"""Deterministic Decimal-only bookkeeping analytics."""

from collections import defaultdict
from decimal import Decimal

from bookkeeping.models import (
    BookkeepingAnalytics,
    BookkeepingTransaction,
    DimensionSummary,
    ExpenseSummary,
    PeriodSummary,
)


class BookkeepingAnalyticsService:
    """Calculate reproducible financial summaries without an LLM."""

    def __init__(self, *, largest_expense_count: int = 10) -> None:
        if largest_expense_count <= 0:
            raise ValueError("largest_expense_count must be positive")
        self._largest_expense_count = largest_expense_count

    def analyze(
        self,
        transactions: tuple[BookkeepingTransaction, ...],
    ) -> BookkeepingAnalytics:
        total_income = sum(
            (item.amount for item in transactions if item.amount > 0),
            Decimal("0"),
        )
        total_expenses = sum(
            (-item.amount for item in transactions if item.amount < 0),
            Decimal("0"),
        )
        net_cash_flow = total_income - total_expenses
        transaction_count = len(transactions)
        average = (
            net_cash_flow / Decimal(transaction_count)
            if transaction_count
            else Decimal("0")
        )

        monthly_values: dict[str, list[Decimal]] = defaultdict(
            lambda: [Decimal("0"), Decimal("0")]
        )
        category_values: dict[str, list[Decimal | int]] = defaultdict(
            lambda: [Decimal("0"), Decimal("0"), 0]
        )
        account_values: dict[str, list[Decimal | int]] = defaultdict(
            lambda: [Decimal("0"), Decimal("0"), 0]
        )
        for transaction in transactions:
            month = transaction.transaction_date.strftime("%Y-%m")
            category = transaction.category or "Uncategorized"
            account = transaction.account_name or "Unspecified"
            income = (
                transaction.amount
                if transaction.amount > 0
                else Decimal("0")
            )
            expense = (
                -transaction.amount
                if transaction.amount < 0
                else Decimal("0")
            )
            monthly_values[month][0] += income
            monthly_values[month][1] += expense
            for values in (category_values[category], account_values[account]):
                values[0] += income
                values[1] += expense
                values[2] += 1

        monthly = tuple(
            PeriodSummary(
                period=month,
                income=values[0],
                expenses=values[1],
                net_cash_flow=values[0] - values[1],
            )
            for month, values in sorted(monthly_values.items())
        )
        categories = self._dimension_summaries(category_values)
        accounts = self._dimension_summaries(account_values)
        expenses = sorted(
            (item for item in transactions if item.amount < 0),
            key=lambda item: (
                item.amount,
                item.transaction_date,
                item.reference,
            ),
        )
        largest_expenses = tuple(
            ExpenseSummary(
                transaction_reference=item.reference,
                transaction_date=item.transaction_date,
                description=item.description,
                amount=-item.amount,
                category=item.category,
                account_name=item.account_name,
            )
            for item in expenses[: self._largest_expense_count]
        )
        uncategorized = tuple(
            item
            for item in transactions
            if not item.category or item.category.casefold() == "uncategorized"
        )
        uncategorized_expenses = sum(
            (-item.amount for item in uncategorized if item.amount < 0),
            Decimal("0"),
        )
        uncategorized_percentage = (
            uncategorized_expenses / total_expenses * Decimal("100")
            if total_expenses
            else Decimal("0")
        )

        return BookkeepingAnalytics(
            total_income=total_income,
            total_expenses=total_expenses,
            net_cash_flow=net_cash_flow,
            transaction_count=transaction_count,
            average_transaction_amount=average,
            monthly=monthly,
            totals_by_category=categories,
            totals_by_account=accounts,
            largest_expenses=largest_expenses,
            uncategorized_transaction_count=len(uncategorized),
            uncategorized_expense_percentage=uncategorized_percentage,
        )

    @staticmethod
    def _dimension_summaries(
        values_by_name: dict[str, list[Decimal | int]],
    ) -> tuple[DimensionSummary, ...]:
        return tuple(
            DimensionSummary(
                name=name,
                income=values[0],
                expenses=values[1],
                net_cash_flow=values[0] - values[1],
                transaction_count=int(values[2]),
            )
            for name, values in sorted(
                values_by_name.items(),
                key=lambda item: item[0].casefold(),
            )
        )
