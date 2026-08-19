"""Bounded local CSV parsing for normalized bookkeeping transactions."""

import csv
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from io import StringIO
import re
from typing import BinaryIO, TextIO

from bookkeeping.config import BookkeepingConfig
from bookkeeping.models import (
    BookkeepingTransaction,
    CSVIngestionResult,
    CSVIngestionSummary,
    CSVRowError,
)


class CSVValidationError(ValueError):
    """Raised when a CSV cannot be safely interpreted as transactions."""


_HEADING_ALIASES = {
    "date": "transaction_date",
    "transaction_date": "transaction_date",
    "posted_date": "transaction_date",
    "description": "description",
    "transaction_description": "description",
    "details": "description",
    "memo": "memo",
    "notes": "memo",
    "payee": "vendor_or_payee",
    "vendor": "vendor_or_payee",
    "vendor_or_payee": "vendor_or_payee",
    "amount": "amount",
    "transaction_amount": "amount",
    "debit": "debit",
    "withdrawal": "debit",
    "withdrawals": "debit",
    "credit": "credit",
    "deposit": "credit",
    "deposits": "credit",
    "category": "category",
    "account": "account_name",
    "account_name": "account_name",
    "transaction_id": "transaction_id",
    "id": "transaction_id",
}
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%m/%d/%y",
    "%Y/%m/%d",
    "%b %d %Y",
    "%B %d %Y",
)
_CURRENCY_CODE = re.compile(
    r"(?i)^(?:USD|CAD|AUD|EUR|GBP)\s*|\s*(?:USD|CAD|AUD|EUR|GBP)$"
)


class BookkeepingCSVLoader:
    """Parse local CSV data without persisting or logging source values."""

    def __init__(self, config: BookkeepingConfig | None = None) -> None:
        self._config = config or BookkeepingConfig()

    def load(
        self,
        source: str | bytes | TextIO | BinaryIO,
    ) -> CSVIngestionResult:
        text = self._read_text(source)
        reader = csv.DictReader(StringIO(text, newline=""))
        if reader.fieldnames is None:
            raise CSVValidationError("CSV must contain a header row")
        heading_map, warnings = self._normalize_headings(reader.fieldnames)
        self._validate_required_headings(heading_map)

        transactions: list[BookkeepingTransaction] = []
        errors: list[CSVRowError] = []
        rows_received = 0
        for row_number, row in enumerate(reader, start=2):
            rows_received += 1
            if rows_received > self._config.maximum_rows:
                raise CSVValidationError(
                    "CSV exceeds configured maximum_rows"
                )
            if None in row:
                errors.append(
                    CSVRowError(
                        row_number,
                        "malformed_row",
                        "Row has more values than the header",
                    )
                )
                continue
            try:
                transaction = self._parse_row(
                    row,
                    heading_map,
                    row_number,
                )
            except ValueError as error:
                errors.append(
                    CSVRowError(
                        row_number,
                        "invalid_transaction",
                        str(error),
                    )
                )
                continue
            transactions.append(transaction)

        summary = CSVIngestionSummary(
            rows_received=rows_received,
            rows_accepted=len(transactions),
            rows_rejected=len(errors),
            warnings=tuple(warnings),
            row_errors=tuple(errors),
        )
        return CSVIngestionResult(tuple(transactions), summary)

    def _read_text(
        self,
        source: str | bytes | TextIO | BinaryIO,
    ) -> str:
        if isinstance(source, str):
            raw: str | bytes = source
        elif isinstance(source, bytes):
            raw = source
        elif hasattr(source, "read"):
            original_position = None
            try:
                if hasattr(source, "tell"):
                    original_position = source.tell()
                if hasattr(source, "seek"):
                    source.seek(0)
                raw = source.read(self._config.maximum_upload_size_bytes + 1)
            except (OSError, TypeError, ValueError) as error:
                raise CSVValidationError(
                    "CSV source could not be read"
                ) from error
            finally:
                if original_position is not None and hasattr(source, "seek"):
                    try:
                        source.seek(original_position)
                    except (OSError, TypeError, ValueError):
                        pass
        else:
            raise TypeError(
                "CSV source must be text, bytes, or a file-like object"
            )

        if isinstance(raw, bytes):
            if len(raw) > self._config.maximum_upload_size_bytes:
                raise CSVValidationError(
                    "CSV exceeds configured maximum_upload_size_bytes"
                )
            try:
                return raw.decode("utf-8-sig")
            except UnicodeDecodeError as error:
                raise CSVValidationError(
                    "CSV must be valid UTF-8 text"
                ) from error
        if not isinstance(raw, str):
            raise CSVValidationError(
                "CSV file-like object must return text or bytes"
            )
        if (
            len(raw.encode("utf-8"))
            > self._config.maximum_upload_size_bytes
        ):
            raise CSVValidationError(
                "CSV exceeds configured maximum_upload_size_bytes"
            )
        return raw.removeprefix("\ufeff")

    @staticmethod
    def _normalize_headings(
        headings: list[str],
    ) -> tuple[dict[str, str], list[str]]:
        canonical_to_source: dict[str, str] = {}
        ignored: list[str] = []
        for heading in headings:
            normalized = normalize_heading(heading)
            canonical = _HEADING_ALIASES.get(normalized)
            if canonical is None:
                ignored.append(normalized or "blank")
                continue
            if canonical in canonical_to_source:
                raise CSVValidationError(
                    f"CSV contains duplicate '{canonical}' headings"
                )
            canonical_to_source[canonical] = heading
        warnings = []
        if ignored:
            warnings.append(
                "Unrecognized columns were ignored: "
                + ", ".join(sorted(set(ignored)))
            )
        if (
            "amount" in canonical_to_source
            and (
                "debit" in canonical_to_source
                or "credit" in canonical_to_source
            )
        ):
            warnings.append(
                "Amount was used when amount and debit/credit columns "
                "were both present"
            )
        return canonical_to_source, warnings

    @staticmethod
    def _validate_required_headings(
        heading_map: dict[str, str],
    ) -> None:
        if "transaction_date" not in heading_map:
            raise CSVValidationError(
                "CSV requires a date, transaction_date, or posted_date column"
            )
        if not {
            "description",
            "memo",
            "vendor_or_payee",
        }.intersection(heading_map):
            raise CSVValidationError(
                "CSV requires a description, memo, payee, or vendor column"
            )
        if "amount" not in heading_map and not {
            "debit",
            "credit",
        }.intersection(heading_map):
            raise CSVValidationError(
                "CSV requires amount or debit/credit columns"
            )

    def _parse_row(
        self,
        row: dict[str, str | None],
        heading_map: dict[str, str],
        row_number: int,
    ) -> BookkeepingTransaction:
        def value(name: str) -> str:
            source_heading = heading_map.get(name)
            raw = row.get(source_heading) if source_heading else None
            return raw.strip() if isinstance(raw, str) else ""

        transaction_date = parse_transaction_date(
            value("transaction_date")
        )
        description = (
            value("description")
            or value("vendor_or_payee")
            or value("memo")
        )
        if not description:
            raise ValueError("Description is required")
        if "amount" in heading_map:
            amount = parse_money(value("amount"))
        else:
            debit_text = value("debit")
            credit_text = value("credit")
            if not debit_text and not credit_text:
                raise ValueError("Amount is required")
            debit = (
                parse_money(debit_text)
                if debit_text
                else Decimal("0")
            )
            credit = (
                parse_money(credit_text)
                if credit_text
                else Decimal("0")
            )
            if debit < 0 or credit < 0:
                raise ValueError(
                    "Debit and credit column values cannot be negative"
                )
            if debit != 0 and credit != 0:
                raise ValueError(
                    "Debit and credit cannot both contain an amount"
                )
            amount = credit - debit

        return BookkeepingTransaction(
            transaction_date=transaction_date,
            description=description,
            amount=amount,
            source_row_number=row_number,
            transaction_id=value("transaction_id") or None,
            account_name=value("account_name") or None,
            category=value("category") or None,
            memo=value("memo") or None,
            vendor_or_payee=value("vendor_or_payee") or None,
        )


def normalize_heading(value: str) -> str:
    """Normalize common CSV heading punctuation and spacing."""

    return re.sub(
        r"_+",
        "_",
        re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()),
    ).strip("_")


def parse_transaction_date(value: str) -> date:
    """Parse supported unambiguous and common US export date formats."""

    normalized = value.strip()
    if not normalized:
        raise ValueError("Transaction date is required")
    for date_format in _DATE_FORMATS:
        try:
            return datetime.strptime(normalized, date_format).date()
        except ValueError:
            continue
    raise ValueError("Transaction date has an unsupported format")


def parse_money(value: str) -> Decimal:
    """Parse an exported monetary value without binary floating point."""

    normalized = value.strip()
    if not normalized:
        raise ValueError("Amount is required")
    parenthesized = normalized.startswith("(") and normalized.endswith(")")
    if normalized.startswith("(") != normalized.endswith(")"):
        raise ValueError("Amount has invalid parentheses")
    if parenthesized:
        normalized = normalized[1:-1].strip()
    normalized = _CURRENCY_CODE.sub("", normalized).strip()
    normalized = normalized.translate(
        str.maketrans("", "", "$\u00a3\u20ac\u00a5, ")
    )
    if parenthesized and normalized.startswith(("+", "-")):
        raise ValueError("Amount has conflicting signs")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", normalized):
        raise ValueError("Amount has an unsupported format")
    try:
        amount = Decimal(normalized)
    except InvalidOperation as error:
        raise ValueError("Amount is invalid") from error
    if not amount.is_finite():
        raise ValueError("Amount must be finite")
    return -amount if parenthesized else amount


def normalized_transactions_csv(
    transactions: tuple[BookkeepingTransaction, ...],
) -> str:
    """Create a deterministic, spreadsheet-safe normalized CSV export."""

    output = StringIO(newline="")
    fieldnames = [
        "transaction_date",
        "description",
        "amount",
        "transaction_id",
        "account_name",
        "category",
        "memo",
        "vendor_or_payee",
    ]
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        lineterminator="\n",
    )
    writer.writeheader()
    for transaction in transactions:
        writer.writerow(
            {
                "transaction_date": transaction.transaction_date.isoformat(),
                "description": spreadsheet_safe_text(
                    transaction.description
                ),
                "amount": str(transaction.amount),
                "transaction_id": spreadsheet_safe_text(
                    transaction.transaction_id or ""
                ),
                "account_name": spreadsheet_safe_text(
                    transaction.account_name or ""
                ),
                "category": spreadsheet_safe_text(
                    transaction.category or ""
                ),
                "memo": spreadsheet_safe_text(transaction.memo or ""),
                "vendor_or_payee": spreadsheet_safe_text(
                    transaction.vendor_or_payee or ""
                ),
            }
        )
    return output.getvalue()


def spreadsheet_safe_text(value: str) -> str:
    """Neutralize text cells that spreadsheet programs may evaluate."""

    if value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value
