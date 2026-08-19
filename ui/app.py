"""Streamlit entrypoint for the local Data Engineering Assistant."""

from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_text = str(PROJECT_ROOT)
if not sys.path or sys.path[0] != project_root_text:
    if project_root_text in sys.path:
        sys.path.remove(project_root_text)
    sys.path.insert(0, project_root_text)

import streamlit as st

from knowledge.application_models import (
    ApplicationRequest,
    ApplicationResponse,
    ApplicationStatus,
    ConversationRole,
)
from ui.bootstrap import (
    RuntimeBundle,
    build_runtime,
    check_local_connections,
    provider_selection_for_mode,
)
from ui.bookkeeping_page import render_bookkeeping_page
from ui.config import (
    EmbeddingProviderName,
    LLMProviderName,
    RuntimeMode,
    UIConfig,
    VALID_UI_ENVIRONMENTS,
    load_ui_config,
    VectorStoreProviderName,
)
from ui.formatting import (
    cost_details,
    response_details,
    safe_error_message,
    source_details,
    source_summary,
    status_presentation,
)
from ui.monitoring_dashboard import render_monitoring_dashboard
from ui.session import (
    FEEDBACK_KEY,
    HISTORY_KEY,
    LAST_RESPONSE_KEY,
    SessionMessage,
    accumulate_response_cost,
    append_message,
    clear_conversation,
    conversation_context,
    ensure_scope,
    feedback_csv,
    feedback_json,
    initialize_session,
    record_feedback,
    session_cost_totals,
)


EXAMPLE_QUESTIONS = (
    "Design an S3-to-Glue-to-Athena pipeline.",
    "Why did my Glue job fail with an access-denied error?",
    "Write a PySpark deduplication transformation.",
    "What information do you need before designing my pipeline?",
    "Deploy my CDK stack.",
    "Delete the production data bucket.",
)

COST_DISCLAIMER = (
    "This is an application estimate, not an AWS invoice. Actual charges "
    "may differ by model, Region, pricing mode, caching, discounts, and AWS "
    "pricing changes."
)


@st.cache_resource(show_spinner=False)
def _cached_runtime(
    config: UIConfig,
    runtime_mode: str,
    client_id: str,
    environment: str,
    top_k: int,
    minimum_similarity: float,
) -> RuntimeBundle:
    return build_runtime(
        config,
        runtime_mode=runtime_mode,
        client_id=client_id,
        environment=environment,
        retrieval_top_k=top_k,
        minimum_similarity=minimum_similarity,
    )


def main() -> None:
    st.set_page_config(
        page_title="AWS Data Engineering Assistant",
        page_icon="🛠️",
        layout="wide",
    )
    st.title("AWS Data Engineering Assistant")
    st.write(
        "A provider-neutral, safety-aware RAG demonstration for data "
        "engineering architecture, code, and troubleshooting."
    )
    page = st.sidebar.radio(
        "Page",
        options=("Assistant", "Bookkeeping", "Offline monitoring"),
        help=(
            "Bookkeeping analyzes an uploaded CSV locally. Monitoring is a "
            "read-only view of committed synthetic evidence."
        ),
    )
    if page == "Bookkeeping":
        render_bookkeeping_page()
        return
    if page == "Offline monitoring":
        render_monitoring_dashboard()
        return
    try:
        config = load_ui_config()
    except Exception as error:
        st.error(safe_error_message(error))
        st.stop()

    initialize_session(st.session_state)
    settings = _render_sidebar(config)
    scope_changed = ensure_scope(
        st.session_state,
        settings["client_id"],
        settings["environment"],
    )
    if scope_changed and st.session_state.get("_scope_initialized"):
        st.info(
            "Conversation history was reset because the client or environment "
            "changed."
        )
    st.session_state["_scope_initialized"] = True

    if settings["llm_provider"] == LLMProviderName.BEDROCK:
        st.warning(
            "Bedrock mode invokes AWS services and may incur cost. Credentials "
            "are loaded only through the standard AWS credential chain and "
            "are never entered or displayed here."
        )
    elif settings["llm_provider"] == LLMProviderName.OLLAMA:
        st.info(
            "Local mode uses host-managed Ollama and Qdrant services. "
            "Hardware, electricity, and hosting costs are not estimated."
        )
    else:
        st.success(
            "Demo mode is offline and uses a deterministic synthetic corpus."
        )

    _render_history(settings)
    _render_examples()
    query = st.text_area(
        "Your request",
        key="query_input",
        height=120,
        placeholder="Ask about an AWS data engineering task.",
        help=(
            "Deployment and destructive requests demonstrate safety controls "
            "and never execute actions."
        ),
    )
    submitted = st.button(
        "Submit request",
        type="primary",
        disabled=not query.strip(),
        width="stretch",
    )
    if submitted:
        _submit(query, config, settings)

    response = st.session_state.get(LAST_RESPONSE_KEY)
    if isinstance(response, ApplicationResponse):
        _render_response(
            response,
            show_debug=settings["show_debug"],
        )
        _render_feedback(response)
    _render_session_totals()
    _render_feedback_export()


def _render_sidebar(config: UIConfig) -> dict[str, object]:
    with st.sidebar:
        st.header("Runtime settings")
        mode = RuntimeMode(
            st.selectbox(
                "Runtime mode",
                options=[mode.value for mode in RuntimeMode],
                index=list(RuntimeMode).index(config.runtime_mode),
                help="Demo is offline. Bedrock uses configured AWS access.",
            )
        )
        llm_provider, embedding_provider, vector_store_provider = (
            provider_selection_for_mode(config, mode)
        )
        st.caption(
            "Providers: "
            f"LLM `{llm_provider.value}` · embeddings "
            f"`{embedding_provider.value}` · vectors "
            f"`{vector_store_provider.value}`"
        )
        if (
            llm_provider == LLMProviderName.OLLAMA
            or embedding_provider == EmbeddingProviderName.OLLAMA
        ):
            st.caption(
                f"Ollama chat `{config.ollama_chat_model}` · embeddings "
                f"`{config.ollama_embedding_model}`"
            )
        if vector_store_provider == VectorStoreProviderName.QDRANT:
            st.caption(f"Qdrant collection `{config.qdrant_collection}`")
        local_selected = (
            llm_provider == LLMProviderName.OLLAMA
            or embedding_provider == EmbeddingProviderName.OLLAMA
            or vector_store_provider == VectorStoreProviderName.QDRANT
        )
        if local_selected and st.button(
            "Test local connections",
            width="stretch",
            help="Checks availability only; it does not run inference.",
        ):
            try:
                statuses = check_local_connections(
                    config,
                    runtime_mode=mode,
                )
                st.session_state["local_connection_status"] = statuses
            except Exception as error:
                st.session_state["local_connection_status"] = {
                    "error": safe_error_message(error)
                }
        status = st.session_state.get("local_connection_status")
        if local_selected and isinstance(status, dict):
            if "error" in status:
                st.error(str(status["error"]))
            else:
                st.success(
                    " · ".join(
                        f"{name}: {value}"
                        for name, value in status.items()
                    )
                )
        client_id = st.text_input(
            "Client ID",
            value=config.default_client_id,
            help="Used as a hard retrieval and conversation scope.",
        ).strip().lower()
        environments = sorted(VALID_UI_ENVIRONMENTS)
        environment = st.selectbox(
            "Environment",
            options=environments,
            index=environments.index(config.default_environment),
        )
        use_history = st.toggle(
            "Include conversation history",
            value=True,
            help="History remains only in this browser session.",
        )
        top_k = st.number_input(
            "Retrieval top-k",
            min_value=1,
            max_value=50,
            value=config.retrieval_top_k,
            step=1,
        )
        minimum_similarity = st.slider(
            "Minimum similarity",
            min_value=-1.0,
            max_value=1.0,
            value=config.minimum_similarity,
            step=0.05,
        )
        show_debug = st.toggle(
            "Show retrieved source summaries",
            value=False,
            help=(
                "Development information only. Full context and embeddings "
                "are never displayed."
            ),
        )
        if st.button("Clear conversation", width="stretch"):
            clear_conversation(st.session_state)
            st.success("Conversation cleared.")
    return {
        "runtime_mode": mode,
        "llm_provider": llm_provider,
        "embedding_provider": embedding_provider,
        "vector_store_provider": vector_store_provider,
        "client_id": client_id,
        "environment": environment,
        "use_history": use_history,
        "top_k": int(top_k),
        "minimum_similarity": float(minimum_similarity),
        "show_debug": show_debug,
        "maximum_conversation_messages": (
            config.maximum_conversation_messages
        ),
        "developer_mode": config.developer_mode,
    }


def _render_examples() -> None:
    st.subheader("Example questions")
    st.caption(
        "The deployment and deletion examples demonstrate non-executing "
        "approval and safety behavior."
    )
    columns = st.columns(2)
    for index, question in enumerate(EXAMPLE_QUESTIONS):
        if columns[index % 2].button(
            question,
            key=f"example_{index}",
            width="stretch",
        ):
            st.session_state["query_input"] = question
            st.rerun()


def _render_history(settings: dict[str, object]) -> None:
    if not settings["use_history"]:
        return
    history = [
        message
        for message in st.session_state[HISTORY_KEY]
        if message.client_id == settings["client_id"]
        and message.environment == settings["environment"]
    ]
    if not history:
        return
    st.subheader("Conversation")
    for message in history:
        with st.chat_message(message.role.value):
            st.markdown(message.content)


def _submit(
    query: str,
    config: UIConfig,
    settings: dict[str, object],
) -> None:
    try:
        conversation = (
            conversation_context(
                st.session_state,
                client_id=str(settings["client_id"]),
                environment=str(settings["environment"]),
                maximum_messages=int(
                    settings["maximum_conversation_messages"]
                ),
            )
            if settings["use_history"]
            else ()
        )
        with st.spinner("Classifying, routing, and preparing a response..."):
            bundle = _cached_runtime(
                config,
                str(settings["runtime_mode"]),
                str(settings["client_id"]),
                str(settings["environment"]),
                int(settings["top_k"]),
                float(settings["minimum_similarity"]),
            )
            request = ApplicationRequest(
                request_id=uuid.uuid4().hex,
                query=query,
                client_id=str(settings["client_id"]),
                environment=str(settings["environment"]),
                conversation_context=conversation,
                metadata={
                    "runtime_mode": str(settings["runtime_mode"]),
                    "sensitive": False,
                    "interface": "streamlit-local",
                },
                timestamp=datetime.now(timezone.utc),
            )
            response = bundle.application.handle(request)
        st.session_state[LAST_RESPONSE_KEY] = response
        accumulate_response_cost(st.session_state, response)
        maximum_messages = int(
            settings["maximum_conversation_messages"]
        )
        append_message(
            st.session_state,
            SessionMessage(
                role=ConversationRole.USER,
                content=query,
                client_id=request.client_id,
                environment=request.environment,
                request_id=request.request_id,
            ),
            maximum_messages=maximum_messages,
        )
        append_message(
            st.session_state,
            SessionMessage(
                role=ConversationRole.ASSISTANT,
                content=response.answer,
                client_id=request.client_id,
                environment=request.environment,
                request_id=request.request_id,
            ),
            maximum_messages=maximum_messages,
        )
    except Exception as error:
        st.error(safe_error_message(error))
        if settings["developer_mode"]:
            st.exception(error)


def _render_response(
    response: ApplicationResponse,
    *,
    show_debug: bool,
) -> None:
    st.subheader("Response")
    severity, heading = status_presentation(response.status)
    getattr(st, severity)(heading)
    st.markdown(response.answer)

    details = response_details(response)
    columns = st.columns(3)
    for index, (label, value) in enumerate(details.items()):
        columns[index % 3].metric(label, value)

    for warning in response.warnings:
        st.warning(warning)

    estimate = response.model_metadata.cost_estimate
    if estimate is not None:
        if not estimate.is_chargeable:
            st.write(
                f"Estimated request cost: {estimate.formatted_total}"
            )
            st.info("Demo mode — no Bedrock charge incurred")
        for warning in estimate.warnings:
            if warning != "Demo mode — no Bedrock charge incurred":
                st.warning(warning)
        with st.expander("Cost details"):
            values = cost_details(estimate)
            for label, value in values.items():
                if (
                    label.startswith("Cache-")
                    and value in {"Not available", "Unavailable"}
                ):
                    continue
                st.write(f"**{label}:** {value}")
            st.caption(COST_DISCLAIMER)

    if response.sources:
        st.subheader("Attributed sources")
        for source in response.sources:
            values = source_details(source)
            with st.expander(
                f"[{source.source_id}] {source.source_name} "
                f"({source.similarity_score:.3f})"
            ):
                st.write(f"Document ID: `{values['document_id']}`")
                st.write(f"Chunk ID: `{values['chunk_id']}`")
                st.write(f"Object key: `{values['object_key']}`")
                if values["page"] is not None:
                    st.write(f"Page: {values['page']}")
                if values["section"] is not None:
                    st.write(f"Section: {values['section']}")
                if values["metadata"]:
                    st.json(values["metadata"])
    if show_debug:
        with st.expander(
            "Development information: retrieved source summaries"
        ):
            st.caption(
                "Summaries contain identifiers and scores only. Full document "
                "content and embeddings are intentionally omitted."
            )
            if response.sources:
                for source in response.sources:
                    st.code(source_summary(source), language=None)
            else:
                st.write("No scoped sources were returned.")
            st.json(
                {
                    "retrieval_attempted": (
                        response.retrieval_metadata.attempted
                    ),
                    "result_count": (
                        response.retrieval_metadata.result_count
                    ),
                    "scope_filtered": (
                        response.retrieval_metadata.filtered_for_scope
                    ),
                    "deduplicated": (
                        response.retrieval_metadata.deduplicated
                    ),
                    "context_characters": (
                        response.retrieval_metadata.context_characters
                    ),
                }
            )


def _render_feedback(response: ApplicationResponse) -> None:
    if response.status != ApplicationStatus.COMPLETED:
        return
    st.subheader("Feedback")
    existing = st.session_state[FEEDBACK_KEY].get(response.request_id)
    if existing is not None:
        st.info("Feedback has already been recorded for this response.")
        return
    comment = st.text_input(
        "Optional short feedback comment",
        max_chars=500,
        key=f"feedback_comment_{response.request_id}",
    )
    positive, negative = st.columns(2)
    rating = None
    if positive.button(
        "👍 Helpful",
        key=f"feedback_up_{response.request_id}",
        width="stretch",
    ):
        rating = "up"
    if negative.button(
        "👎 Not helpful",
        key=f"feedback_down_{response.request_id}",
        width="stretch",
    ):
        rating = "down"
    if rating is not None:
        if record_feedback(
            st.session_state,
            request_id=response.request_id,
            rating=rating,
            comment=comment,
            response=response,
        ):
            st.success("Feedback saved in this session.")
        else:
            st.info("Feedback was already recorded for this response.")


def _render_session_totals() -> None:
    totals = session_cost_totals(st.session_state)
    st.subheader("Current session estimates")
    columns = st.columns(4)
    columns[0].metric("Requests in current session", totals.request_count)
    columns[1].metric("Total input tokens", totals.total_input_tokens)
    columns[2].metric("Total output tokens", totals.total_output_tokens)
    columns[3].metric(
        "Total estimated session cost",
        f"${totals.total_estimated_cost:.6f}",
    )
    st.caption(
        "Session-only totals are not persisted. Demo requests may contribute "
        "token counts, but never AWS charges."
    )


def _render_feedback_export() -> None:
    if not st.session_state[FEEDBACK_KEY]:
        return
    st.subheader("Session feedback export")
    export_format = st.radio(
        "Export format",
        options=("JSON", "CSV"),
        horizontal=True,
    )
    if export_format == "JSON":
        payload = feedback_json(st.session_state)
        filename = "data-engineering-assistant-feedback.json"
        mime = "application/json"
    else:
        payload = feedback_csv(st.session_state)
        filename = "data-engineering-assistant-feedback.csv"
        mime = "text/csv"
    st.download_button(
        "Download current-session feedback",
        data=payload,
        file_name=filename,
        mime=mime,
    )


if __name__ == "__main__":
    main()
