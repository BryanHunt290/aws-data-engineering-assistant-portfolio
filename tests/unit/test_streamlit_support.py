from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from knowledge.application_models import (
    ApplicationRequest,
    ApplicationResponse,
    ApplicationStatus,
    ConversationRole,
    ModelMetadata,
    RetrievalMetadata,
    SourceCitation,
)
from knowledge.embedding_errors import EmbeddingThrottledError
from knowledge.intents import Intent
from knowledge.llm_errors import LLMAccessDeniedError
from knowledge.routing import Route
from ui.bootstrap import (
    DEMO_LICENSE,
    DeterministicDemoEmbeddingProvider,
    build_runtime,
    provider_selection_for_mode,
    load_demo_documents,
)
from ui.config import (
    EmbeddingProviderName,
    LLMProviderName,
    RuntimeMode,
    UIConfig,
    VectorStoreProviderName,
    load_ui_config,
)
from ui.formatting import (
    response_details,
    safe_error_message,
    source_details,
    source_summary,
    status_presentation,
)
from ui.session import (
    FEEDBACK_KEY,
    HISTORY_KEY,
    LAST_RESPONSE_KEY,
    FeedbackRecord,
    SessionMessage,
    append_message,
    clear_conversation,
    conversation_context,
    ensure_scope,
    feedback_csv,
    feedback_json,
    initialize_session,
    record_feedback,
)


FIXED_TIME = datetime(2026, 7, 27, 20, 0, tzinfo=timezone.utc)


def test_streamlit_entrypoint_prioritizes_repository_package_imports():
    repository = Path.cwd().resolve()
    ui_directory = repository / "ui"
    entrypoint = ui_directory / "app.py"
    code = (
        "import runpy, sys; "
        f"sys.path.insert(0, {str(ui_directory)!r}); "
        f"runpy.run_path({str(entrypoint)!r}, run_name='streamlit_smoke'); "
        "from knowledge.vector_store import normalize_vector_scope; "
        "print(normalize_vector_scope('Demo Client', 'dev'))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "('demo-client', 'dev')" in completed.stdout


def _response(
    *,
    status: ApplicationStatus = ApplicationStatus.COMPLETED,
) -> ApplicationResponse:
    source = SourceCitation(
        source_id="S1",
        document_id="demo-doc",
        chunk_id="demo-doc:000000",
        source_name="Synthetic Guide",
        object_key="demo://synthetic/guide.md",
        similarity_score=0.87654,
        page=2,
        section="Testing",
        metadata={
            "client_id": "client-a",
            "environment": "dev",
            "topic": "testing",
            "synthetic": True,
            "license": DEMO_LICENSE,
            "secret": "must-not-display",
            "embedding_vector": [0.1, 0.2],
        },
    )
    return ApplicationResponse(
        request_id="request-1",
        answer="Grounded answer [S1].",
        intent=Intent.KNOWLEDGE_QUESTION,
        route=Route.RETRIEVAL,
        confidence=0.875,
        sources=(source,),
        retrieval_metadata=RetrievalMetadata(
            attempted=True,
            result_count=1,
            requested_top_k=5,
            minimum_similarity=0.2,
            context_characters=120,
        ),
        model_metadata=ModelMetadata(
            provider_name="deterministic-fake",
            model_id="fake-llm",
            input_token_count=25,
            output_token_count=5,
            finish_reason="end_turn",
            latency_ms=1.5,
        ),
        approval_required=False,
        safety_review_required=False,
        latency_ms=4.5,
        warnings=("Synthetic response.",),
        status=status,
    )


def test_ui_configuration_defaults_to_offline_demo():
    config = load_ui_config({})

    assert config.runtime_mode == RuntimeMode.DEMO
    assert config.aws_region == "us-west-2"
    assert config.default_client_id == "demo-client"
    assert config.default_environment == "dev"
    assert config.retrieval_top_k == 5
    assert config.minimum_similarity == 0.0
    assert config.maximum_conversation_messages == 10
    assert config.developer_mode is False
    assert config.llm_provider == LLMProviderName.FAKE
    assert config.embedding_provider == EmbeddingProviderName.FAKE
    assert config.vector_store_provider == VectorStoreProviderName.MEMORY


def test_ui_configuration_selects_local_ollama_and_qdrant_profile():
    config = load_ui_config({"APP_RUNTIME_MODE": "local"})

    assert config.runtime_mode == RuntimeMode.LOCAL
    assert config.llm_provider == LLMProviderName.OLLAMA
    assert config.embedding_provider == EmbeddingProviderName.OLLAMA
    assert config.vector_store_provider == VectorStoreProviderName.QDRANT
    assert config.ollama_url == "http://localhost:11434"
    assert config.ollama_embedding_model == "embeddinggemma"
    assert config.ollama_chat_model == "qwen3:8b"
    assert config.qdrant_url == "http://localhost:6333"
    assert config.qdrant_collection == "dea_knowledge_embeddinggemma_v1"


def test_explicit_provider_environment_is_independent_of_runtime_profile():
    config = load_ui_config(
        {
            "APP_RUNTIME_MODE": "local",
            "LLM_PROVIDER": "fake",
            "EMBEDDING_PROVIDER": "fake",
            "VECTOR_STORE_PROVIDER": "memory",
        }
    )

    assert provider_selection_for_mode(config, RuntimeMode.LOCAL) == (
        LLMProviderName.FAKE,
        EmbeddingProviderName.FAKE,
        VectorStoreProviderName.MEMORY,
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LLM_PROVIDER", "unknown"),
        ("EMBEDDING_PROVIDER", "unknown"),
        ("VECTOR_STORE_PROVIDER", "unknown"),
    ],
)
def test_ui_configuration_rejects_unsupported_provider(name, value):
    with pytest.raises(ValueError, match="provider"):
        load_ui_config({name: value})


def test_ui_configuration_rejects_missing_selected_local_settings():
    with pytest.raises(ValueError, match="Ollama"):
        load_ui_config(
            {},
            overrides={"llm_provider": "ollama", "ollama_url": ""},
        )


def test_ui_configuration_loads_environment_and_local_overrides():
    config = load_ui_config(
        {
            "DEA_RUNTIME_MODE": "bedrock",
            "DEA_AWS_REGION": "US-EAST-1",
            "DEA_EMBEDDING_MODEL_ID": "embed-model",
            "DEA_LLM_MODEL_ID": "llm-model",
            "DEA_DEFAULT_CLIENT_ID": "client-a",
            "DEA_DEFAULT_ENVIRONMENT": "stage",
            "DEA_RETRIEVAL_TOP_K": "8",
            "DEA_MINIMUM_SIMILARITY": "0.35",
            "DEA_MAXIMUM_CONVERSATION_MESSAGES": "6",
            "DEA_DEVELOPER_MODE": "yes",
            "DEA_PRICING_CATALOG_PATH": "config/prices.json",
        },
        overrides={"retrieval_top_k": 9},
    )

    assert config.runtime_mode == RuntimeMode.BEDROCK
    assert config.aws_region == "us-east-1"
    assert config.embedding_model_id == "embed-model"
    assert config.llm_model_id == "llm-model"
    assert config.default_client_id == "client-a"
    assert config.default_environment == "stage"
    assert config.retrieval_top_k == 9
    assert config.minimum_similarity == 0.35
    assert config.maximum_conversation_messages == 6
    assert config.developer_mode is True
    assert config.pricing_catalog_path == "config/prices.json"


def test_container_environment_names_are_supported_and_take_precedence():
    config = load_ui_config(
        {
            "APP_RUNTIME_MODE": "demo",
            "DEA_RUNTIME_MODE": "bedrock",
            "AWS_REGION": "us-east-2",
            "DEA_AWS_REGION": "us-west-1",
            "APP_EMBEDDING_MODEL_ID": "container-embedding",
            "APP_LLM_MODEL_ID": "container-llm",
            "APP_DEFAULT_CLIENT_ID": "container-client",
            "APP_DEFAULT_ENVIRONMENT": "test",
            "APP_RETRIEVAL_TOP_K": "7",
            "APP_MINIMUM_SIMILARITY": "0.25",
            "APP_MAXIMUM_CONVERSATION_MESSAGES": "4",
            "APP_DEVELOPER_MODE": "false",
        }
    )

    assert config.runtime_mode == RuntimeMode.DEMO
    assert config.aws_region == "us-east-2"
    assert config.embedding_model_id == "container-embedding"
    assert config.llm_model_id == "container-llm"
    assert config.default_client_id == "container-client"
    assert config.default_environment == "test"
    assert config.retrieval_top_k == 7
    assert config.minimum_similarity == 0.25
    assert config.maximum_conversation_messages == 4
    assert config.developer_mode is False


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"DEA_RUNTIME_MODE": "invalid"}, "runtime_mode"),
        ({"DEA_RETRIEVAL_TOP_K": "none"}, "integer"),
        ({"DEA_MINIMUM_SIMILARITY": "none"}, "number"),
        ({"DEA_DEVELOPER_MODE": "perhaps"}, "true or false"),
        ({"DEA_DEFAULT_CLIENT_ID": "bad_client"}, "default_client_id"),
        ({"DEA_DEFAULT_ENVIRONMENT": "qa"}, "default_environment"),
    ],
)
def test_ui_configuration_rejects_invalid_values(environment, message):
    with pytest.raises(ValueError, match=message):
        load_ui_config(environment)


def test_synthetic_demo_corpus_is_complete_reproducible_and_labeled():
    first = load_demo_documents()
    second = load_demo_documents()

    assert first == second
    assert len(first) == 7
    assert {document.topic for document in first} == {
        "athena-query-troubleshooting",
        "cost-awareness",
        "glue-job-troubleshooting",
        "iam-least-privilege",
        "pipeline-monitoring",
        "pyspark-transformations",
        "s3-data-lake-architecture",
    }
    assert all(
        "License: CC0-1.0" in document.text
        and "synthetic demonstration content" in document.text
        for document in first
    )
    assert all("demo://synthetic/" in document.object_key for document in first)


def test_demo_bootstrap_and_request_are_fully_offline(monkeypatch):
    def reject_aws_call(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Demo mode must not construct an AWS client")

    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(client=reject_aws_call),
    )
    config = UIConfig(
        runtime_mode=RuntimeMode.DEMO,
        default_client_id="client-a",
        default_environment="dev",
        minimum_similarity=-1.0,
    )

    bundle = build_runtime(config)
    response = bundle.application.handle(
        ApplicationRequest(
            request_id="offline-demo",
            query=(
                "Why did my Glue job fail with an access-denied error?"
            ),
            client_id="client-a",
            environment="dev",
            timestamp=FIXED_TIME,
        )
    )

    assert bundle.runtime_mode == RuntimeMode.DEMO
    assert bundle.corpus_document_count == 7
    assert bundle.corpus_chunk_count >= 7
    assert bundle.embedding_provider_name == "deterministic-demo"
    assert bundle.llm_provider_name == "deterministic-fake"
    assert response.status == ApplicationStatus.COMPLETED
    assert response.model_metadata.cost_estimate is not None
    assert (
        response.model_metadata.cost_estimate.formatted_total
        == "$0.000000"
    )
    assert response.model_metadata.cost_estimate.is_chargeable is False
    assert response.sources
    assert all(
        source.metadata["client_id"] == "client-a"
        and source.metadata["environment"] == "dev"
        for source in response.sources
    )


def test_demo_bootstrap_does_not_initialize_ollama_or_qdrant(monkeypatch):
    def reject_local_initialization(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Unselected local providers must stay lazy")

    monkeypatch.setattr(
        "ui.bootstrap.OllamaEmbeddingProvider",
        reject_local_initialization,
    )
    monkeypatch.setattr(
        "ui.bootstrap.OllamaLLMProvider",
        reject_local_initialization,
    )
    monkeypatch.setattr(
        "ui.bootstrap.QdrantVectorStore",
        reject_local_initialization,
    )

    bundle = build_runtime(UIConfig())

    assert bundle.embedding_provider_name == "deterministic-demo"
    assert bundle.llm_provider_name == "deterministic-fake"
    assert bundle.vector_store_provider_name == "memory"


def test_demo_bootstrap_safety_examples_execute_nothing(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(
            client=lambda *args, **kwargs: (
                _ for _ in ()
            ).throw(AssertionError("No AWS call expected"))
        ),
    )
    bundle = build_runtime(
        UIConfig(
            default_client_id="client-a",
            default_environment="dev",
        )
    )

    deployment = bundle.application.handle(
        ApplicationRequest(
            "deploy-demo",
            "Deploy my CDK stack.",
            "client-a",
            "dev",
            timestamp=FIXED_TIME,
        )
    )
    deletion = bundle.application.handle(
        ApplicationRequest(
            "delete-demo",
            "Delete the production data bucket.",
            "client-a",
            "dev",
            timestamp=FIXED_TIME,
        )
    )

    assert deployment.status == ApplicationStatus.APPROVAL_REQUIRED
    assert deployment.approval_required is True
    assert deletion.status == ApplicationStatus.SAFETY_REVIEW_REQUIRED
    assert deletion.safety_review_required is True


def test_bedrock_bootstrap_uses_only_injected_runtime_client():
    client = Mock()
    client.invoke_model.side_effect = lambda **kwargs: {
        "body": BytesIO(b'{"embedding": [1.0, 0.0]}')
    }
    client.converse.return_value = {
        "output": {
            "message": {
                "content": [{"text": "Mocked Bedrock answer [S1]."}]
            }
        },
        "usage": {"inputTokens": 10, "outputTokens": 5},
        "stopReason": "end_turn",
    }
    config = UIConfig(
        runtime_mode=RuntimeMode.BEDROCK,
        embedding_model_id="embedding-model",
        llm_model_id="llm-model",
        default_client_id="client-a",
        default_environment="dev",
        minimum_similarity=-1.0,
    )

    bundle = build_runtime(
        config,
        bedrock_runtime_client=client,
    )
    response = bundle.application.handle(
        ApplicationRequest(
            "bedrock-mock",
            "Find this in the runbook",
            "client-a",
            "dev",
            timestamp=FIXED_TIME,
        )
    )

    assert bundle.runtime_mode == RuntimeMode.BEDROCK
    assert bundle.embedding_provider_name == "amazon-bedrock"
    assert bundle.llm_provider_name == "amazon-bedrock"
    assert response.status == ApplicationStatus.COMPLETED
    assert response.model_metadata.model_id == "llm-model"
    assert response.model_metadata.input_token_count == 10
    assert response.model_metadata.output_token_count == 5
    assert response.model_metadata.cost_estimate is not None
    assert response.model_metadata.cost_estimate.available is False
    assert client.invoke_model.call_count > bundle.corpus_document_count
    client.converse.assert_called_once()


def test_demo_embedding_is_stable_and_keyword_sensitive():
    provider = DeterministicDemoEmbeddingProvider(dimensions=16)

    first = provider.embed(["glue access denied"])[0]
    repeated = provider.embed(["glue access denied"])[0]
    different = provider.embed(["pyspark deduplication"])[0]

    assert first == repeated
    assert first != different
    assert len(first) == 16


def test_ui_entrypoint_does_not_import_or_call_backend_components_directly():
    app_source = (
        Path(__file__).parents[2] / "ui" / "app.py"
    ).read_text(encoding="utf-8")

    for forbidden in (
        "BedrockLLMProvider",
        "BedrockEmbeddingProvider",
        "InMemoryCosineRetriever",
        "RuleBasedIntentClassifier",
        "RequestRouter(",
        ".retrieve(",
        ".classify(",
        ".embed(",
    ):
        assert forbidden not in app_source
    assert "bundle.application.handle(request)" in app_source


def test_session_initialization_and_scope_reset_isolated():
    state = {}

    initialize_session(state)
    assert state[HISTORY_KEY] == []
    assert state[FEEDBACK_KEY] == {}
    assert ensure_scope(state, "client-a", "dev") is True
    append_message(
        state,
        SessionMessage(
            ConversationRole.USER,
            "client A",
            "client-a",
            "dev",
        ),
        maximum_messages=4,
    )
    state[LAST_RESPONSE_KEY] = _response()
    state[FEEDBACK_KEY]["request-1"] = FeedbackRecord(
        "request-1",
        "up",
        "",
        "2026-07-27T20:00:00Z",
    )

    assert ensure_scope(state, "client-b", "dev") is True
    assert state[HISTORY_KEY] == []
    assert state[LAST_RESPONSE_KEY] is None
    assert "request-1" in state[FEEDBACK_KEY]
    assert ensure_scope(state, "client-b", "dev") is False


def test_conversation_history_is_bounded_and_preserves_roles():
    state = {}
    ensure_scope(state, "client-a", "dev")
    for index, role in enumerate(
        (
            ConversationRole.USER,
            ConversationRole.ASSISTANT,
            ConversationRole.USER,
        )
    ):
        append_message(
            state,
            SessionMessage(
                role,
                f"message-{index}",
                "client-a",
                "dev",
            ),
            maximum_messages=2,
        )

    context = conversation_context(
        state,
        client_id="client-a",
        environment="dev",
        maximum_messages=2,
    )

    assert [message.content for message in context] == [
        "message-1",
        "message-2",
    ]
    assert [message.role for message in context] == [
        ConversationRole.ASSISTANT,
        ConversationRole.USER,
    ]
    assert all(message.client_id == "client-a" for message in context)


def test_session_rejects_message_for_another_scope():
    state = {}
    ensure_scope(state, "client-a", "dev")

    with pytest.raises(ValueError, match="scope"):
        append_message(
            state,
            SessionMessage(
                ConversationRole.USER,
                "wrong",
                "client-b",
                "dev",
            ),
            maximum_messages=2,
        )


def test_clear_conversation_does_not_clear_feedback():
    state = {}
    ensure_scope(state, "client-a", "dev")
    append_message(
        state,
        SessionMessage(
            ConversationRole.USER,
            "message",
            "client-a",
            "dev",
        ),
        maximum_messages=2,
    )
    record_feedback(
        state,
        request_id="request-1",
        rating="up",
        created_at=FIXED_TIME,
    )

    clear_conversation(state)

    assert state[HISTORY_KEY] == []
    assert "request-1" in state[FEEDBACK_KEY]


def test_response_and_status_formatting_use_backend_terminology():
    response = _response()

    details = response_details(response)

    assert details["Status"] == "completed"
    assert details["Intent"] == "knowledge_question"
    assert details["Route"] == "retrieval"
    assert details["Classifier confidence"] == "87.5%"
    assert details["Approval required"] == "No"
    assert details["Model ID"] == "fake-llm"
    assert details["Input tokens"] == "25"
    assert status_presentation(ApplicationStatus.COMPLETED) == (
        "success",
        "Completed",
    )
    assert status_presentation(
        ApplicationStatus.SAFETY_REVIEW_REQUIRED
    ) == ("error", "Safety review required")


def test_source_formatting_omits_embeddings_and_unselected_metadata():
    source = _response().sources[0]

    details = source_details(source)
    summary = source_summary(source)
    serialized = json.dumps(details)

    assert details["source_name"] == "Synthetic Guide"
    assert details["document_id"] == "demo-doc"
    assert details["chunk_id"] == "demo-doc:000000"
    assert details["similarity_score"] == 0.8765
    assert details["page"] == 2
    assert details["section"] == "Testing"
    assert details["object_key"] == "demo://synthetic/guide.md"
    assert details["metadata"]["license"] == DEMO_LICENSE
    assert "secret" not in serialized
    assert "embedding_vector" not in serialized
    assert "document text" not in summary


def test_feedback_is_deduplicated_and_comment_is_bounded():
    state = {}

    accepted = record_feedback(
        state,
        request_id="request-1",
        rating="up",
        comment="x" * 700,
        created_at=FIXED_TIME,
    )
    duplicate = record_feedback(
        state,
        request_id="request-1",
        rating="down",
        created_at=FIXED_TIME,
    )

    assert accepted is True
    assert duplicate is False
    record = state[FEEDBACK_KEY]["request-1"]
    assert record.rating == "up"
    assert len(record.comment) == 500


def test_feedback_json_and_csv_export_current_session():
    state = {}
    record_feedback(
        state,
        request_id="request-2",
        rating="down",
        comment="Needs sources",
        created_at=FIXED_TIME,
    )
    record_feedback(
        state,
        request_id="request-1",
        rating="up",
        comment="Useful",
        created_at=FIXED_TIME,
    )

    json_payload = json.loads(feedback_json(state))
    csv_payload = feedback_csv(state)

    assert [item["request_id"] for item in json_payload] == [
        "request-1",
        "request-2",
    ]
    assert csv_payload.splitlines()[0].startswith(
        "request_id,rating,comment,created_at,model_id,input_tokens,"
    )
    assert "request-1,up,Useful,2026-07-27T20:00:00Z" in csv_payload
    assert "request-2,down,Needs sources,2026-07-27T20:00:00Z" in csv_payload


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            LLMAccessDeniedError("private"),
            "Bedrock access was denied",
        ),
        (
            EmbeddingThrottledError("private"),
            "throttling requests",
        ),
        (
            ValueError("bad top-k"),
            "Configuration error: bad top-k",
        ),
        (
            RuntimeError("private stack trace"),
            "could not be completed safely",
        ),
    ],
)
def test_safe_error_formatting_hides_internal_details(error, expected):
    message = safe_error_message(error)

    assert expected in message
    assert "private" not in message
