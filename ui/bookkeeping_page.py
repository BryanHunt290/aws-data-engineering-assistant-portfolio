"""Private, local Streamlit presentation for bookkeeping analysis."""

from decimal import Decimal
import hashlib
import math
import re

import streamlit as st

from bookkeeping.analytics import BookkeepingAnalyticsService
from bookkeeping.categorization import AdvisoryCategorizationService
from bookkeeping.config import (
    BookkeepingLLMProvider,
    load_bookkeeping_config,
)
from bookkeeping.csv_loader import (
    BookkeepingCSVLoader,
    CSVValidationError,
    normalized_transactions_csv,
)
from bookkeeping.duplicates import DuplicateDetector
from bookkeeping.grounded_answers import GroundedBookkeepingAnswerService
from bookkeeping.knowledge_models import (
    BookkeepingAuthorityLevel,
    BookkeepingDocumentType,
    BookkeepingKnowledgeMetadata,
    BookkeepingMetadataValidationError,
    BookkeepingRetrievalMode,
    GroundedBookkeepingAnswer,
)
from bookkeeping.knowledge_service import (
    BookkeepingKnowledgeService,
    format_bookkeeping_citation,
)
from bookkeeping.models import (
    BookkeepingAnalytics,
    BookkeepingTransaction,
    CategorySuggestion,
    DuplicateCandidate,
)
from bookkeeping.providers import build_bookkeeping_llm_provider
from bookkeeping.reporting import BusinessReportService
from knowledge.fake_embeddings import DeterministicFakeEmbeddingProvider
from knowledge.llm_errors import LLMProviderError
from knowledge.pdf_extraction import PdfExtractionError


_FILE_DIGEST_KEY = "bookkeeping_file_digest"
_SUGGESTIONS_KEY = "bookkeeping_suggestions"
_REPORT_KEY = "bookkeeping_report"
_KNOWLEDGE_SERVICE_KEY = "bookkeeping_knowledge_service"
_KNOWLEDGE_ANSWER_KEY = "bookkeeping_knowledge_answer"
_KNOWLEDGE_CLIENT_KEY = "bookkeeping_knowledge_client_id"


def render_bookkeeping_page() -> None:
    """Render local analysis; invoke a model only after an explicit click."""

    st.header("Private bookkeeping analysis")
    st.warning(
        "Category suggestions and AI explanations require human review. "
        "This tool does not provide accounting or tax advice."
    )
    st.caption(
        "CSV bytes remain in this Streamlit session. This phase does not "
        "upload transactions to S3, QuickBooks, Bedrock, or another service."
    )
    try:
        config = load_bookkeeping_config()
    except ValueError:
        st.error(
            "Bookkeeping configuration is invalid. Review the documented "
            "DEA_* environment variables."
        )
        return

    knowledge_service = _knowledge_service(config)
    _render_bookkeeping_knowledge(knowledge_service, config)
    st.divider()
    st.subheader("Transaction analysis")

    upload_limit_mb = max(
        1,
        math.ceil(config.maximum_upload_size_bytes / (1024 * 1024)),
    )
    uploaded = st.file_uploader(
        "Upload an exported transaction CSV",
        type="csv",
        key="bookkeeping_csv",
        max_upload_size=upload_limit_mb,
        help=(
            f"Local analysis only; maximum {upload_limit_mb} MiB and "
            f"{config.maximum_rows:,} rows."
        ),
    )
    if uploaded is None:
        st.info(
            "Upload a CSV or review the synthetic example at "
            "`data/bookkeeping/sample_transactions.csv`."
        )
        return

    content = uploaded.getvalue()
    try:
        ingestion = BookkeepingCSVLoader(config).load(content)
    except (CSVValidationError, TypeError, ValueError):
        st.error(
            "The CSV could not be validated. Check its headings, UTF-8 "
            "encoding, row values, and configured limits."
        )
        return

    digest = hashlib.sha256(content).hexdigest()
    if st.session_state.get(_FILE_DIGEST_KEY) != digest:
        st.session_state[_FILE_DIGEST_KEY] = digest
        st.session_state.pop(_SUGGESTIONS_KEY, None)
        st.session_state.pop(_REPORT_KEY, None)

    _render_ingestion_summary(ingestion.summary)
    transactions = ingestion.transactions
    if not transactions:
        st.error("No valid transactions were available for analysis.")
        return

    st.subheader("Normalized transaction preview")
    st.dataframe(
        transaction_preview_rows(transactions),
        hide_index=True,
        key="bookkeeping_transaction_preview",
    )
    st.download_button(
        "Download normalized CSV",
        data=normalized_transactions_csv(transactions),
        file_name="normalized-bookkeeping-transactions.csv",
        mime="text/csv",
        help=(
            "Text cells are protected against spreadsheet-formula injection."
        ),
    )

    analytics = BookkeepingAnalyticsService().analyze(transactions)
    duplicates = DuplicateDetector(
        date_window_days=config.duplicate_date_window_days
    ).detect(transactions)
    _render_analytics(analytics, duplicates, transactions)

    st.subheader("Advisory model actions")
    st.info(
        f"Configured provider: `{config.llm_provider.value}`. No provider is "
        "called until you approve and select an action."
    )
    if config.llm_provider == BookkeepingLLMProvider.BEDROCK:
        st.warning(
            "Bedrock is not enabled for financial rows in this local phase. "
            "Select `fake` or `ollama`; there is no automatic fallback."
        )
    approved = st.checkbox(
        "I approve sending transaction descriptions and amounts to the "
        "configured provider for this action.",
        key="bookkeeping_llm_approval",
        help=(
            "Account names, memos, transaction IDs, and the original CSV are "
            "not sent. Ollama must use a loopback URL."
        ),
    )
    use_policy_context = st.checkbox(
        "Use approved bookkeeping policy context for category suggestions",
        key="bookkeeping_use_policy_context",
        help=(
            "Only active, approved general references and references matching "
            "the current client ID are eligible."
        ),
    )
    include_report_references = st.checkbox(
        "Include approved reference context in the business summary",
        key="bookkeeping_include_report_references",
    )
    provider_allowed = (
        config.llm_provider != BookkeepingLLMProvider.BEDROCK
    )
    actions = st.container(horizontal=True)
    with actions:
        suggest_clicked = st.button(
            "Request category suggestions",
            icon=":material/category:",
            disabled=not approved or not provider_allowed,
            key="bookkeeping_suggest",
        )
        report_clicked = st.button(
            "Generate business summary",
            icon=":material/description:",
            key="bookkeeping_report",
        )

    if suggest_clicked:
        with st.spinner("Requesting advisory category suggestions..."):
            try:
                provider = build_bookkeeping_llm_provider(config)
                suggestions = AdvisoryCategorizationService(
                    provider,
                    batch_size=config.categorization_batch_size,
                    knowledge_service=knowledge_service,
                ).suggest(
                    transactions,
                    use_policy_context=use_policy_context,
                    client_id=(
                        st.session_state.get(_KNOWLEDGE_CLIENT_KEY) or None
                    ),
                    retrieval_mode=st.session_state.get(
                        "bookkeeping_knowledge_retrieval_mode",
                        BookkeepingRetrievalMode.KEYWORD,
                    ),
                )
                st.session_state[_SUGGESTIONS_KEY] = suggestions
            except Exception:
                st.error(
                    "Category suggestions are unavailable. Confirm the local "
                    "provider configuration and try again."
                )

    suggestions = st.session_state.get(_SUGGESTIONS_KEY)
    if isinstance(suggestions, tuple):
        if any(
            item.source == "deterministic-fallback"
            for item in suggestions
        ):
            st.warning(
                "Some model results were unavailable or invalid. "
                "Deterministic fallback suggestions are shown instead."
            )
        st.dataframe(
            suggestion_rows(suggestions),
            hide_index=True,
            key="bookkeeping_suggestions_table",
        )
        st.caption(
            "These are suggestions only. Existing categories were not changed."
        )

    if report_clicked:
        include_ai = approved and provider_allowed
        try:
            provider = (
                build_bookkeeping_llm_provider(config)
                if include_ai
                else None
            )
            grounded_service = (
                GroundedBookkeepingAnswerService(
                    knowledge_service=knowledge_service,
                    llm_provider=provider,
                )
                if provider is not None and include_report_references
                else None
            )
            report = BusinessReportService(
                duplicate_detector=DuplicateDetector(
                    date_window_days=config.duplicate_date_window_days
                ),
                llm_provider=provider,
                grounded_answer_service=grounded_service,
            ).generate(
                transactions,
                include_ai_explanation=include_ai,
                include_reference_context=include_report_references,
                client_id=(
                    st.session_state.get(_KNOWLEDGE_CLIENT_KEY) or None
                ),
                retrieval_mode=st.session_state.get(
                    "bookkeeping_knowledge_retrieval_mode",
                    BookkeepingRetrievalMode.KEYWORD,
                ),
                category_suggestions=(
                    suggestions if isinstance(suggestions, tuple) else ()
                ),
            )
            st.session_state[_REPORT_KEY] = report
        except Exception:
            st.error(
                "The business summary could not be generated safely."
            )

    report = st.session_state.get(_REPORT_KEY)
    if report is not None:
        st.subheader("Business summary report")
        for warning in report.warnings:
            st.warning(warning)
        st.markdown(report.markdown)
        st.download_button(
            "Download Markdown report",
            data=report.markdown,
            file_name="bookkeeping-business-summary.md",
            mime="text/markdown",
        )


def _knowledge_service(config) -> BookkeepingKnowledgeService:
    service = st.session_state.get(_KNOWLEDGE_SERVICE_KEY)
    if isinstance(service, BookkeepingKnowledgeService):
        return service
    service = BookkeepingKnowledgeService(
        embedding_provider=DeterministicFakeEmbeddingProvider(
            model_id="bookkeeping-session-embedding-v1",
            dimensions=32,
        ),
        chunk_size=config.knowledge_chunk_size,
        overlap=config.knowledge_chunk_overlap,
        maximum_upload_size=config.maximum_upload_size_bytes,
        maximum_passages=config.knowledge_maximum_passages,
        maximum_context_characters=(
            config.knowledge_maximum_context_characters
        ),
    )
    st.session_state[_KNOWLEDGE_SERVICE_KEY] = service
    return service


def _render_bookkeeping_knowledge(
    service: BookkeepingKnowledgeService,
    config,
) -> None:
    st.subheader("Bookkeeping knowledge")
    st.caption(
        "Upload text-based PDFs as supporting references. Scanned, image-only, "
        "and encrypted PDFs are unsupported; OCR is not enabled. Uploading and "
        "ingestion do not call an LLM."
    )
    with st.form("bookkeeping_knowledge_ingestion"):
        pdf = st.file_uploader(
            "Bookkeeping reference PDF",
            type="pdf",
            key="bookkeeping_reference_pdf",
            max_upload_size=max(
                1,
                math.ceil(config.maximum_upload_size_bytes / (1024 * 1024)),
            ),
        )
        title = st.text_input(
            "Document title",
            key="bookkeeping_reference_title",
            max_chars=200,
        )
        document_type = st.selectbox(
            "Document type",
            options=tuple(BookkeepingDocumentType),
            format_func=lambda value: value.value.replace("_", " "),
            key="bookkeeping_reference_type",
        )
        authority = st.selectbox(
            "Authority level (optional)",
            options=(None, *tuple(BookkeepingAuthorityLevel)),
            format_func=lambda value: (
                "Not specified"
                if value is None
                else value.value.replace("_", " ")
            ),
            key="bookkeeping_reference_authority",
        )
        client_specific = st.checkbox(
            "This reference is client-specific",
            key="bookkeeping_reference_client_specific",
        )
        client_id = st.text_input(
            "Client ID (required for client-specific references)",
            key="bookkeeping_reference_client_id",
            max_chars=64,
            help=(
                "Leave blank for an explicitly general, non-client-specific "
                "reference."
            ),
        )
        with st.container(horizontal=True):
            effective_date = st.date_input(
                "Effective date (optional)",
                value=None,
                key="bookkeeping_reference_effective_date",
            )
            review_date = st.date_input(
                "Review date (optional)",
                value=None,
                key="bookkeeping_reference_review_date",
            )
        approved = st.checkbox(
            "Approved for bookkeeping context",
            key="bookkeeping_reference_approved",
            help=(
                "A reference is never eligible for bookkeeping retrieval "
                "without this explicit approval."
            ),
        )
        ingest_clicked = st.form_submit_button(
            "Ingest approved reference",
            icon=":material/upload_file:",
        )

    if ingest_clicked:
        if pdf is None:
            st.error("Choose a text-based PDF before ingestion.")
        elif not approved:
            st.error(
                "Explicit bookkeeping-context approval is required before "
                "this reference can be used."
            )
        else:
            try:
                metadata = BookkeepingKnowledgeMetadata(
                    document_type=document_type,
                    title=title,
                    source_filename=pdf.name,
                    approved_for_bookkeeping=approved,
                    client_specific=client_specific,
                    client_id=client_id or None,
                    effective_date=effective_date,
                    review_date=review_date,
                    authority_level=authority,
                )
                with st.spinner("Extracting, chunking, and embedding locally..."):
                    document = service.ingest_pdf(
                        filename=pdf.name,
                        content=pdf.getvalue(),
                        metadata=metadata,
                    )
                st.success(
                    "Reference ingested: "
                    f"{document.page_count or 'unknown'} page(s), "
                    f"{document.chunk_count} chunk(s), embedding status "
                    f"{document.embedding_status}."
                )
            except (
                BookkeepingMetadataValidationError,
                PdfExtractionError,
                TypeError,
                ValueError,
            ):
                st.error(
                    "The reference could not be ingested safely. Check the "
                    "metadata and confirm the PDF contains extractable text."
                )

    visible_client = st.text_input(
        "Client ID for reference list and questions (optional)",
        key=_KNOWLEDGE_CLIENT_KEY,
        max_chars=64,
    )
    try:
        documents = service.list_documents(
            client_id=visible_client or None,
        )
    except Exception:
        documents = ()
        st.error("The client scope is invalid.")
    st.markdown("#### Currently approved bookkeeping references")
    active_documents = tuple(
        document
        for document in documents
        if document.classification.approved_for_bookkeeping
        and document.classification.active
    )
    if active_documents:
        st.dataframe(
            approved_reference_rows(active_documents),
            hide_index=True,
            key="bookkeeping_approved_references",
        )
        selection = st.selectbox(
            "Reference to exclude from future retrieval",
            options=(None, *active_documents),
            format_func=lambda document: (
                "Select a reference"
                if document is None
                else document.classification.title
            ),
            key="bookkeeping_deactivate_reference",
        )
        if st.button(
            "Exclude selected reference",
            icon=":material/block:",
            disabled=selection is None,
            key="bookkeeping_deactivate_reference_button",
        ):
            service.deactivate(selection.document_id)
            st.session_state.pop(_KNOWLEDGE_ANSWER_KEY, None)
            st.success(
                "The reference was deactivated without deleting its content."
            )
            st.rerun()
    else:
        st.info("No active approved references are visible in this scope.")

    with st.form("bookkeeping_knowledge_question"):
        question = st.text_area(
            "Ask a bookkeeping question using approved references",
            key="bookkeeping_knowledge_question_text",
            max_chars=2_000,
            placeholder=(
                "How should software subscriptions be categorized under our "
                "approved policy?"
            ),
        )
        retrieval_mode = st.selectbox(
            "Retrieval mode",
            options=tuple(BookkeepingRetrievalMode),
            format_func=lambda value: value.value,
            key="bookkeeping_knowledge_retrieval_mode",
        )
        generation_approved = st.checkbox(
            "I approve sending only the retrieved passages and question to "
            "the configured provider for this action.",
            key="bookkeeping_knowledge_generation_approval",
        )
        ask_clicked = st.form_submit_button(
            "Retrieve context and generate answer",
            icon=":material/search:",
            disabled=(
                not generation_approved
                or config.llm_provider == BookkeepingLLMProvider.BEDROCK
            ),
        )
    if config.llm_provider == BookkeepingLLMProvider.BEDROCK:
        st.warning(
            "Bedrock remains selectable in configuration but is disabled for "
            "this local bookkeeping workflow. There is no cloud fallback."
        )
    if ask_clicked:
        try:
            provider = build_bookkeeping_llm_provider(config)
            answer = GroundedBookkeepingAnswerService(
                knowledge_service=service,
                llm_provider=provider,
            ).answer(
                question,
                client_id=visible_client or None,
                retrieval_mode=retrieval_mode,
            )
            st.session_state[_KNOWLEDGE_ANSWER_KEY] = answer
        except LLMProviderError:
            st.error(
                "The selected local model is unavailable. Confirm Ollama is "
                "running and the configured model is installed."
            )
        except Exception:
            st.error(
                "The grounded answer could not be generated safely. No cloud "
                "provider was used as a fallback."
            )

    answer = st.session_state.get(_KNOWLEDGE_ANSWER_KEY)
    if isinstance(answer, GroundedBookkeepingAnswer):
        _render_grounded_answer(answer)


def _render_grounded_answer(answer: GroundedBookkeepingAnswer) -> None:
    st.markdown("#### Grounded bookkeeping answer")
    if not answer.relevant_context_found:
        st.warning("No relevant approved bookkeeping context was found.")
    if answer.sources_conflict:
        st.warning("Retrieved sources conflict and require human review.")
    for warning in answer.warnings:
        st.warning(warning)
    st.markdown(answer.answer_text)
    if answer.retrieved_passages:
        st.markdown("##### Retrieved-source preview")
        for passage in answer.retrieved_passages:
            with st.expander(
                format_bookkeeping_citation(passage.citation)
            ):
                st.write(passage.text)
                st.caption(
                    f"Retrieval score: "
                    f"{passage.citation.retrieval_score:.6f}"
                )
    if answer.retrieved_citations:
        st.markdown("##### Citations")
        for citation in answer.retrieved_citations:
            st.write(format_bookkeeping_citation(citation))
    st.caption(
        f"Provider: {answer.provider_name}; model: "
        f"{answer.model_name or 'not invoked'}; retrieval: "
        f"{answer.retrieval_mode.value}."
    )
    st.warning(
        "Human review is required. This output is not accounting, tax, legal, "
        "audit, or CPA advice."
    )


def approved_reference_rows(documents) -> list[dict[str, object]]:
    return [
        {
            "Title": document.classification.title,
            "Filename": document.classification.source_filename,
            "Type": document.classification.document_type.value,
            "Client": document.classification.client_id or "General",
            "Pages": document.page_count or "Unavailable",
            "Chunks": document.chunk_count,
            "Embedding": document.embedding_status,
            "Active": document.classification.active,
        }
        for document in documents
    ]


def _render_ingestion_summary(summary) -> None:
    st.subheader("Validation summary")
    with st.container(horizontal=True):
        st.metric("Rows received", summary.rows_received, border=True)
        st.metric("Rows accepted", summary.rows_accepted, border=True)
        st.metric("Rows rejected", summary.rows_rejected, border=True)
    for warning in summary.warnings:
        st.warning(warning)
    if summary.row_errors:
        st.dataframe(
            [
                {
                    "Row": error.row_number,
                    "Error": error.error_code,
                    "Explanation": error.message,
                }
                for error in summary.row_errors
            ],
            hide_index=True,
            key="bookkeeping_validation_errors",
        )


def _render_analytics(
    analytics: BookkeepingAnalytics,
    duplicates: tuple[DuplicateCandidate, ...],
    transactions: tuple[BookkeepingTransaction, ...],
) -> None:
    st.subheader("Deterministic analysis")
    with st.container(horizontal=True):
        st.metric(
            "Total income",
            money_text(analytics.total_income),
            border=True,
        )
        st.metric(
            "Total expenses",
            money_text(analytics.total_expenses),
            border=True,
        )
        st.metric(
            "Net cash flow",
            money_text(analytics.net_cash_flow),
            border=True,
        )
        st.metric(
            "Transactions",
            analytics.transaction_count,
            border=True,
        )
    st.caption(
        "All calculations use Decimal. Expenses are displayed as positive "
        "magnitudes; net cash flow retains its sign."
    )

    monthly, categories, accounts = st.tabs(
        ["Monthly summary", "Category totals", "Account totals"]
    )
    with monthly:
        st.dataframe(
            monthly_rows(analytics),
            hide_index=True,
            key="bookkeeping_monthly",
        )
    with categories:
        st.dataframe(
            dimension_rows(analytics.totals_by_category),
            hide_index=True,
            key="bookkeeping_categories",
        )
    with accounts:
        st.dataframe(
            dimension_rows(
                analytics.totals_by_account,
                redact_names=True,
            ),
            hide_index=True,
            key="bookkeeping_accounts",
        )

    st.subheader("Likely duplicates")
    if duplicates:
        st.dataframe(
            duplicate_rows(duplicates),
            hide_index=True,
            key="bookkeeping_duplicates",
        )
        st.caption(
            "Candidates are never deleted or merged automatically."
        )
    else:
        st.success("No likely duplicate pairs were detected.")

    st.subheader("Uncategorized transactions")
    uncategorized = tuple(
        item
        for item in transactions
        if not item.category or item.category.casefold() == "uncategorized"
    )
    st.write(
        f"{analytics.uncategorized_transaction_count} transactions; "
        f"{analytics.uncategorized_expense_percentage:.2f}% of expense "
        "dollars."
    )
    if uncategorized:
        st.dataframe(
            transaction_preview_rows(uncategorized),
            hide_index=True,
            key="bookkeeping_uncategorized",
        )


def transaction_preview_rows(
    transactions: tuple[BookkeepingTransaction, ...],
) -> list[dict[str, object]]:
    """Select deliberate browser-safe columns from normalized transactions."""

    return [
        {
            "Reference": item.reference,
            "Date": item.transaction_date,
            "Description": item.description,
            "Amount": money_text(item.amount),
            "Category": item.category or "Uncategorized",
            "Account": redact_account_name(item.account_name),
        }
        for item in transactions
    ]


def monthly_rows(analytics: BookkeepingAnalytics) -> list[dict[str, str]]:
    return [
        {
            "Month": item.period,
            "Income": money_text(item.income),
            "Expenses": money_text(item.expenses),
            "Net cash flow": money_text(item.net_cash_flow),
        }
        for item in analytics.monthly
    ]


def dimension_rows(
    dimensions,
    *,
    redact_names: bool = False,
) -> list[dict[str, object]]:
    return [
        {
            "Name": (
                redact_account_name(item.name)
                if redact_names
                else item.name
            ),
            "Income": money_text(item.income),
            "Expenses": money_text(item.expenses),
            "Net cash flow": money_text(item.net_cash_flow),
            "Transactions": item.transaction_count,
        }
        for item in dimensions
    ]


def duplicate_rows(
    duplicates: tuple[DuplicateCandidate, ...],
) -> list[dict[str, str]]:
    return [
        {
            "First": item.first_reference,
            "Second": item.second_reference,
            "Rule": item.rule.value,
            "Confidence": f"{item.confidence:.0%}",
            "Explanation": item.explanation,
        }
        for item in duplicates
    ]


def suggestion_rows(
    suggestions: tuple[CategorySuggestion, ...],
) -> list[dict[str, str]]:
    return [
        {
            "Reference": item.transaction_reference,
            "Suggested category": item.suggested_category,
            "Confidence": f"{item.confidence:.0%}",
            "Rationale": item.rationale,
            "Source": item.source,
            "Review required": "Yes",
            "Policy context": item.context_basis,
            "Policy citations": ", ".join(
                f"[{citation.citation_id}]"
                for citation in item.supporting_citations
            ) or "None",
            "Policy conflict": "Yes" if item.policy_conflict else "No",
        }
        for item in suggestions
    ]


def money_text(value: Decimal) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def redact_account_name(value: str | None) -> str:
    if not value:
        return "Unspecified"
    return re.sub(r"\d{4,}", "[redacted]", value)
