from datetime import date
from decimal import Decimal
from io import BytesIO, StringIO
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from bookkeeping.analytics import BookkeepingAnalyticsService
from bookkeeping.categorization import AdvisoryCategorizationService
from bookkeeping.config import (
    BookkeepingConfig,
    BookkeepingLLMProvider,
    load_bookkeeping_config,
)
from bookkeeping.csv_loader import (
    BookkeepingCSVLoader,
    CSVValidationError,
    normalized_transactions_csv,
    normalize_heading,
    parse_money,
    parse_transaction_date,
)
from bookkeeping.duplicates import DuplicateDetector
from bookkeeping.models import BookkeepingTransaction, DuplicateRule
from bookkeeping.providers import build_bookkeeping_llm_provider
from bookkeeping.reporting import BusinessReportService
from knowledge.bedrock_llm import BedrockLLMProvider
from knowledge.fake_llm import DeterministicFakeLLMProvider
from knowledge.llm_errors import LLMModelUnavailableError
from knowledge.ollama_llm import OllamaLLMProvider
from ui.bookkeeping_page import (
    money_text,
    redact_account_name,
    transaction_preview_rows,
)


def _transaction(
    row: int,
    amount: str,
    *,
    transaction_date: date = date(2026, 1, 1),
    description: str = "Synthetic transaction",
    transaction_id: str | None = None,
    account_name: str | None = None,
    category: str | None = None,
    memo: str | None = None,
) -> BookkeepingTransaction:
    return BookkeepingTransaction(
        transaction_date=transaction_date,
        description=description,
        amount=Decimal(amount),
        source_row_number=row,
        transaction_id=transaction_id,
        account_name=account_name,
        category=category,
        memo=memo,
    )


def test_heading_normalization_and_common_aliases():
    assert normalize_heading(" Posted Date ") == "posted_date"
    csv_text = (
        "Posted Date,Payee,Transaction Amount,ID,Account Name\n"
        "01/02/2026,Synthetic Client,$1,TX-1,Checking\n"
    )

    result = BookkeepingCSVLoader().load(csv_text)

    assert result.summary.rows_accepted == 1
    transaction = result.transactions[0]
    assert transaction.transaction_date == date(2026, 1, 2)
    assert transaction.description == "Synthetic Client"
    assert transaction.amount == Decimal("1")
    assert transaction.transaction_id == "TX-1"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$1,234.56", Decimal("1234.56")),
        ("(45.67)", Decimal("-45.67")),
        ("-$9.25", Decimal("-9.25")),
        ("USD 2,000.00", Decimal("2000.00")),
        ("€12.50", Decimal("12.50")),
    ],
)
def test_amount_parsing(raw, expected):
    assert parse_money(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-02-03", date(2026, 2, 3)),
        ("02/03/2026", date(2026, 2, 3)),
        ("2/3/26", date(2026, 2, 3)),
        ("Feb 03 2026", date(2026, 2, 3)),
    ],
)
def test_date_parsing(raw, expected):
    assert parse_transaction_date(raw) == expected


def test_debit_and_credit_are_converted_to_signed_amounts():
    result = BookkeepingCSVLoader().load(
        "date,memo,debit,credit\n"
        "2026-01-01,Synthetic rent,1200,\n"
        "2026-01-02,Synthetic receipt,,2500\n"
    )

    assert [item.amount for item in result.transactions] == [
        Decimal("-1200"),
        Decimal("2500"),
    ]

    missing = BookkeepingCSVLoader().load(
        "date,memo,debit,credit\n"
        "2026-01-03,Synthetic missing amount,,\n"
    )
    assert missing.summary.rows_rejected == 1
    assert missing.summary.row_errors[0].message == "Amount is required"


def test_invalid_rows_are_reported_without_source_values():
    result = BookkeepingCSVLoader().load(
        "date,description,amount\n"
        "bad-date,PRIVATE ACCOUNT 123456789,$999\n"
        "2026-01-01,Synthetic valid,25\n"
    )

    assert result.summary.rows_received == 2
    assert result.summary.rows_accepted == 1
    assert result.summary.rows_rejected == 1
    error = result.summary.row_errors[0]
    assert error.row_number == 2
    assert error.error_code == "invalid_transaction"
    assert "PRIVATE" not in error.message
    assert "123456789" not in error.message
    assert "$999" not in error.message


def test_file_limits_and_file_position_are_enforced():
    stream = BytesIO(
        b"date,description,amount\n2026-01-01,Synthetic,1\n"
    )
    stream.seek(4)
    result = BookkeepingCSVLoader().load(stream)
    assert result.summary.rows_accepted == 1
    assert stream.tell() == 4

    tiny = BookkeepingConfig(maximum_upload_size_bytes=5)
    with pytest.raises(CSVValidationError, match="maximum_upload"):
        BookkeepingCSVLoader(tiny).load("date,description,amount\n")

    one_row = BookkeepingConfig(maximum_rows=1)
    with pytest.raises(CSVValidationError, match="maximum_rows"):
        BookkeepingCSVLoader(one_row).load(
            "date,description,amount\n"
            "2026-01-01,One,1\n"
            "2026-01-02,Two,2\n"
        )


def test_normalized_export_neutralizes_spreadsheet_formulas():
    transaction = _transaction(
        2,
        "-10",
        description="=WEBSERVICE(\"https://example.invalid\")",
        account_name="+CMD",
        memo="@formula",
    )

    exported = normalized_transactions_csv((transaction,))

    assert "'=WEBSERVICE" in exported
    assert "'+CMD" in exported
    assert "'@formula" in exported


def test_analytics_are_decimal_deterministic_and_grouped():
    transactions = (
        _transaction(
            2,
            "1000.00",
            category="Income",
            account_name="Checking",
        ),
        _transaction(
            3,
            "-200.00",
            category="Software",
            account_name="Card",
        ),
        _transaction(
            4,
            "-100.00",
            transaction_date=date(2026, 2, 1),
            account_name="Card",
        ),
    )

    result = BookkeepingAnalyticsService().analyze(transactions)

    assert result.total_income == Decimal("1000.00")
    assert result.total_expenses == Decimal("300.00")
    assert result.net_cash_flow == Decimal("700.00")
    assert result.transaction_count == 3
    assert result.average_transaction_amount == Decimal("700.00") / 3
    assert [(row.period, row.net_cash_flow) for row in result.monthly] == [
        ("2026-01", Decimal("800.00")),
        ("2026-02", Decimal("-100.00")),
    ]
    category = {row.name: row for row in result.totals_by_category}
    assert category["Software"].expenses == Decimal("200.00")
    assert category["Uncategorized"].expenses == Decimal("100.00")
    account = {row.name: row for row in result.totals_by_account}
    assert account["Card"].expenses == Decimal("300.00")
    assert result.largest_expenses[0].amount == Decimal("200.00")
    assert result.uncategorized_transaction_count == 1
    assert result.uncategorized_expense_percentage == Decimal(
        "33.33333333333333333333333333"
    )


def test_duplicate_detection_is_explainable_and_non_mutating():
    transactions = (
        _transaction(2, "-25", transaction_id="same", description="Cafe"),
        _transaction(
            3,
            "-99",
            transaction_id="SAME",
            description="Different",
        ),
        _transaction(4, "-10", description="Software, Inc."),
        _transaction(5, "-10", description="software inc"),
        _transaction(
            6,
            "-10",
            transaction_date=date(2026, 1, 3),
            description="SOFTWARE INC",
        ),
    )
    original = tuple(transactions)

    results = DuplicateDetector(date_window_days=3).detect(transactions)

    rules = {item.rule for item in results}
    assert DuplicateRule.IDENTICAL_TRANSACTION_ID in rules
    assert DuplicateRule.SAME_DESCRIPTION_AMOUNT_DATE in rules
    assert DuplicateRule.SAME_DESCRIPTION_AMOUNT_NEAR_DATE in rules
    assert all(item.explanation for item in results)
    assert transactions == original
    assert len(transactions) == 5


def test_categorization_validates_json_and_preserves_existing_categories():
    response = json.dumps(
        {
            "suggestions": [
                {
                    "reference": "row-2",
                    "category": "software",
                    "confidence": 0.9,
                    "rationale": "The description indicates a subscription.",
                }
            ]
        }
    )
    provider = DeterministicFakeLLMProvider(response_text=response)
    transactions = (
        _transaction(
            2,
            "-50",
            description="Synthetic software subscription",
            account_name="PRIVATE 123456",
            memo="private memo",
        ),
        _transaction(
            3,
            "-100",
            description="Existing",
            category="Rent",
        ),
    )

    suggestions = AdvisoryCategorizationService(provider).suggest(
        transactions
    )

    assert len(suggestions) == 1
    assert suggestions[0].suggested_category == "software"
    assert suggestions[0].requires_review is True
    assert suggestions[0].source == "deterministic-fake"
    prompt = provider.calls[0]["user_prompt"]
    assert "Synthetic software subscription" in prompt
    assert "PRIVATE 123456" not in prompt
    assert "private memo" not in prompt
    assert "Existing" not in prompt
    assert transactions[1].category == "Rent"


@pytest.mark.parametrize(
    "provider",
    [
        DeterministicFakeLLMProvider(response_text="not-json"),
        DeterministicFakeLLMProvider(
            simulated_error=LLMModelUnavailableError("private")
        ),
        DeterministicFakeLLMProvider(
            response_text=json.dumps(
                {
                    "suggestions": [
                        {
                            "reference": "row-2",
                            "category": "software",
                            "confidence": 0.9,
                            "rationale": "This is tax-deductible.",
                        }
                    ]
                }
            )
        ),
    ],
)
def test_invalid_or_unavailable_categorization_uses_safe_fallback(provider):
    suggestion = AdvisoryCategorizationService(provider).suggest(
        (
            _transaction(
                2,
                "-25",
                description="Synthetic software subscription",
            ),
        )
    )[0]

    assert suggestion.suggested_category == "software"
    assert suggestion.source == "deterministic-fallback"
    assert suggestion.requires_review is True


def test_report_keeps_deterministic_totals_separate_from_ai_output():
    provider = DeterministicFakeLLMProvider(
        response_text="Review the unusually high software spending."
    )
    transactions = (
        _transaction(2, "1000", category="Income"),
        _transaction(3, "-200", category="Software"),
    )

    report = BusinessReportService(llm_provider=provider).generate(
        transactions,
        include_ai_explanation=True,
    )

    assert report.analytics.net_cash_flow == Decimal("800")
    assert "Total income: $1,000.00" in report.markdown
    assert "Total expenses: $200.00" in report.markdown
    assert "AI-generated explanation" in report.markdown
    assert report.ai_explanation == (
        "Review the unusually high software spending."
    )
    prompt = provider.calls[0]["user_prompt"]
    assert "Synthetic transaction" not in prompt


def test_report_discards_prohibited_tax_claims_but_keeps_calculations():
    provider = DeterministicFakeLLMProvider(
        response_text="This expense is tax-deductible."
    )
    report = BusinessReportService(llm_provider=provider).generate(
        (_transaction(2, "-25"),),
        include_ai_explanation=True,
    )

    assert report.ai_explanation is None
    assert report.analytics.total_expenses == Decimal("25")
    assert "AI explanation was unavailable" in report.warnings[0]


def test_provider_configuration_is_explicit_and_never_falls_back():
    fake = DeterministicFakeLLMProvider()
    assert (
        build_bookkeeping_llm_provider(
            BookkeepingConfig(
                llm_provider=BookkeepingLLMProvider.FAKE
            ),
            fake_provider=fake,
        )
        is fake
    )
    ollama = build_bookkeeping_llm_provider(
        BookkeepingConfig(llm_provider=BookkeepingLLMProvider.OLLAMA),
        http_session=Mock(),
    )
    assert isinstance(ollama, OllamaLLMProvider)
    bedrock_client = Mock()
    bedrock = build_bookkeeping_llm_provider(
        BookkeepingConfig(llm_provider=BookkeepingLLMProvider.BEDROCK),
        bedrock_runtime_client=bedrock_client,
    )
    assert isinstance(bedrock, BedrockLLMProvider)
    bedrock_client.converse.assert_not_called()


def test_environment_configuration_and_validation():
    config = load_bookkeeping_config(
        {
            "DEA_LLM_PROVIDER": "ollama",
            "DEA_OLLAMA_BASE_URL": "http://127.0.0.1:11434/",
            "DEA_OLLAMA_MODEL": "gpt-oss:20b",
            "DEA_OLLAMA_TIMEOUT_SECONDS": "30",
            "DEA_BOOKKEEPING_MAX_ROWS": "50",
            "DEA_BOOKKEEPING_KNOWLEDGE_MAX_PASSAGES": "3",
            "DEA_BOOKKEEPING_KNOWLEDGE_MAX_CONTEXT_CHARACTERS": "4000",
        }
    )
    assert config.llm_provider == BookkeepingLLMProvider.OLLAMA
    assert config.ollama_base_url == "http://127.0.0.1:11434"
    assert config.ollama_read_timeout_seconds == 30.0
    assert config.maximum_rows == 50
    assert config.knowledge_maximum_passages == 3
    assert config.knowledge_maximum_context_characters == 4000

    with pytest.raises(ValueError):
        load_bookkeeping_config({"DEA_LLM_PROVIDER": "automatic"})


def test_streamlit_helpers_format_money_and_redact_account_digits():
    transactions = (
        _transaction(
            2,
            "-5.5",
            account_name="Checking 123456789",
        ),
    )
    rows = transaction_preview_rows(transactions)

    assert money_text(Decimal("-5.5")) == "-$5.50"
    assert redact_account_name("Checking 123456789") == (
        "Checking [redacted]"
    )
    assert "123456789" not in rows[0]["Account"]


def test_reviewed_sample_is_synthetic_and_loadable():
    path = Path("data/bookkeeping/sample_transactions.csv")
    content = path.read_text(encoding="utf-8")

    assert "Synthetic" in content
    assert "@" not in content
    result = BookkeepingCSVLoader().load(StringIO(content))
    assert result.summary.rows_received >= 10
    assert result.summary.rows_rejected == 0


def test_streamlit_bookkeeping_page_waits_for_explicit_upload_and_action():
    app = AppTest.from_file("ui/app.py", default_timeout=20)

    app.run()
    app.radio[0].set_value("Bookkeeping").run()

    assert not app.error
    assert any(
        "Private bookkeeping analysis" in heading.value
        for heading in app.header
    )
    assert any(
        "not provide accounting or tax advice" in warning.value
        for warning in app.warning
    )
    labels = {button.label: button for button in app.button}
    assert "Ingest approved reference" in labels
    assert labels["Retrieve context and generate answer"].disabled
    assert not any(
        "Grounded bookkeeping answer" in heading.value
        for heading in app.markdown
    )
