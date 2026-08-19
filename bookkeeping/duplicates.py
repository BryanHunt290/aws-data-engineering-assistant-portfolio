"""Explainable, non-mutating duplicate transaction detection."""

from decimal import Decimal
import re
import unicodedata

from bookkeeping.models import (
    BookkeepingTransaction,
    DuplicateCandidate,
    DuplicateRule,
)


class DuplicateDetector:
    """Flag likely duplicates without deleting or merging anything."""

    def __init__(self, *, date_window_days: int = 3) -> None:
        if date_window_days < 0:
            raise ValueError("date_window_days cannot be negative")
        self._date_window_days = date_window_days

    def detect(
        self,
        transactions: tuple[BookkeepingTransaction, ...],
    ) -> tuple[DuplicateCandidate, ...]:
        candidates: list[DuplicateCandidate] = []
        for index, first in enumerate(transactions):
            for second in transactions[index + 1 :]:
                match = self._match(first, second)
                if match is not None:
                    candidates.append(match)
        return tuple(
            sorted(
                candidates,
                key=lambda item: (
                    -item.confidence,
                    item.first_reference,
                    item.second_reference,
                ),
            )
        )

    def _match(
        self,
        first: BookkeepingTransaction,
        second: BookkeepingTransaction,
    ) -> DuplicateCandidate | None:
        if (
            first.transaction_id
            and second.transaction_id
            and first.transaction_id.casefold()
            == second.transaction_id.casefold()
        ):
            return DuplicateCandidate(
                first.reference,
                second.reference,
                DuplicateRule.IDENTICAL_TRANSACTION_ID,
                Decimal("1.00"),
                "The two transactions share the same transaction ID.",
            )
        if (
            _normalized_description(first.description)
            != _normalized_description(second.description)
            or first.amount != second.amount
        ):
            return None
        day_difference = abs(
            (first.transaction_date - second.transaction_date).days
        )
        if day_difference == 0:
            return DuplicateCandidate(
                first.reference,
                second.reference,
                DuplicateRule.SAME_DESCRIPTION_AMOUNT_DATE,
                Decimal("0.95"),
                "Description, amount, and transaction date are identical.",
            )
        if day_difference <= self._date_window_days:
            return DuplicateCandidate(
                first.reference,
                second.reference,
                DuplicateRule.SAME_DESCRIPTION_AMOUNT_NEAR_DATE,
                Decimal("0.80"),
                "Description and amount match within the configured date "
                "window.",
            )
        return None


def _normalized_description(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", normalized))
