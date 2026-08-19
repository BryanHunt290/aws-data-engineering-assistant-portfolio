from datetime import datetime, timezone
import json
import logging
from typing import Any, Mapping, Sequence
from unittest.mock import Mock

import pytest

from knowledge.application import RAGApplicationService
from knowledge.application_models import (
    ApplicationRequest,
    ApplicationStatus,
    ConversationMessage,
    ConversationRole,
)
from knowledge.bedrock_llm import BedrockLLMProvider
from knowledge.classification import RuleBasedIntentClassifier
from knowledge.config import ApplicationConfig
from knowledge.costs import CatalogCostEstimator
from knowledge.embedding_errors import EmbeddingThrottledError
from knowledge.fake_embeddings import DeterministicFakeEmbeddingProvider
from knowledge.fake_llm import DeterministicFakeLLMProvider
from knowledge.intents import Intent
from knowledge.llm import GenerationResult, LLMProvider
from knowledge.llm_errors import (
    LLMAccessDeniedError,
    LLMInvocationError,
    LLMModelUnavailableError,
    LLMThrottledError,
    LLMValidationError,
    MalformedLLMResponseError,
)
from knowledge.prompting import GroundedPromptBuilder, PromptBuilder
from knowledge.rag_evaluation import (
    RAGEvaluationCase,
    RAGEvaluator,
)
from knowledge.retrieval import RetrievalResult
from knowledge.routing import RequestRouter, Route


FIXED_TIME = datetime(2026, 7, 27, 18, 0, tzinfo=timezone.utc)


class StubRetriever:
    def __init__(
        self,
        results: Sequence[RetrievalResult] = (),
        *,
        error: Exception | None = None,
    ) -> None:
        self.results = list(results)
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int | None = None,
        minimum_similarity: float | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append(
            {
                "query_vector": tuple(query_vector),
                "top_k": top_k,
                "minimum_similarity": minimum_similarity,
            }
        )
        if self.error is not None:
            raise self.error
        return list(self.results)


class FailingClassifier:
    classifier_version = "failure-v1"

    def classify(self, query: str, **kwargs):
        del query, kwargs
        raise RuntimeError("private classifier details")


class FailingRouter:
    def route(self, classification, **kwargs):
        del classification, kwargs
        raise RuntimeError("private router details")


class FailingPromptBuilder:
    prompt_version = "grounded-rag-v1"

    def build(self, **kwargs):
        del kwargs
        raise RuntimeError("private prompt details")


class InvalidLLMProvider:
    provider_name = "invalid"

    def __init__(self, result: Any) -> None:
        self.result = result

    def generate(self, **kwargs):
        del kwargs
        return self.result


class FakeServiceError(Exception):
    def __init__(self, code: str) -> None:
        self.response = {"Error": {"Code": code}}
        super().__init__(code)


def _result(
    *,
    document_id: str = "doc-1",
    chunk_id: str = "chunk-1",
    text: str = "Glue retries use exponential backoff.",
    score: float = 0.9,
    client_id: str = "client-a",
    environment: str = "dev",
    source: str = "operations-runbook",
    metadata: Mapping[str, Any] | None = None,
) -> RetrievalResult:
    values = {
        "client_id": client_id,
        "environment": environment,
        "object_key": (
            f"knowledge/raw/{client_id}/{environment}/{document_id}.md"
        ),
        "page": 4,
        "section": "Retries",
    }
    values.update(metadata or {})
    return RetrievalResult(
        document_id=document_id,
        chunk_id=chunk_id,
        source=source,
        text=text,
        similarity_score=score,
        metadata=values,
    )


def _request(
    query: str,
    *,
    request_id: str = "request-1",
    client_id: str = "client-a",
    environment: str = "dev",
    conversation_context: tuple[ConversationMessage, ...] = (),
    metadata: Mapping[str, Any] | None = None,
) -> ApplicationRequest:
    return ApplicationRequest(
        request_id=request_id,
        query=query,
        client_id=client_id,
        environment=environment,
        conversation_context=conversation_context,
        metadata=metadata or {},
        timestamp=FIXED_TIME,
    )


def _application(
    *,
    retriever: StubRetriever | None = None,
    llm_provider: Any | None = None,
    embedding_provider: Any | None = None,
    classifier: Any | None = None,
    router: Any | None = None,
    prompt_builder: Any | None = None,
    config: ApplicationConfig | None = None,
    event_logger: logging.Logger | None = None,
    cost_estimator=None,
    runtime_mode: str = "bedrock",
) -> RAGApplicationService:
    selected_config = config or ApplicationConfig(minimum_similarity=0.2)
    return RAGApplicationService(
        classifier=classifier or RuleBasedIntentClassifier(),
        router=router or RequestRouter(),
        embedding_provider=embedding_provider
        or DeterministicFakeEmbeddingProvider(dimensions=4),
        retriever=retriever or StubRetriever(),
        prompt_builder=prompt_builder
        or GroundedPromptBuilder(
            prompt_version=selected_config.prompt_version
        ),
        llm_provider=llm_provider
        or DeterministicFakeLLMProvider(
            response_text="Use exponential backoff [S1]."
        ),
        config=selected_config,
        event_logger=event_logger,
        cost_estimator=cost_estimator,
        runtime_mode=runtime_mode,
    )


def test_successful_retrieval_grounded_response_and_source_attribution():
    retriever = StubRetriever([_result()])
    llm = DeterministicFakeLLMProvider(
        response_text="Use exponential backoff [S1].",
        provider_metadata={"trace": "fake-trace"},
    )

    response = _application(
        retriever=retriever,
        llm_provider=llm,
    ).handle(_request("Find retry guidance in the runbook"))

    assert response.status == ApplicationStatus.COMPLETED
    assert response.intent == Intent.KNOWLEDGE_QUESTION
    assert response.route == Route.RETRIEVAL
    assert response.answer == "Use exponential backoff [S1]."
    assert response.retrieval_metadata.result_count == 1
    assert response.model_metadata.model_id == "fake-llm-v1"
    assert response.model_metadata.provider_metadata == {
        "trace": "fake-trace"
    }
    citation = response.sources[0]
    assert citation.source_id == "S1"
    assert citation.document_id == "doc-1"
    assert citation.chunk_id == "chunk-1"
    assert citation.source_name == "operations-runbook"
    assert citation.object_key.endswith("doc-1.md")
    assert citation.page == 4
    assert citation.section == "Retries"
    assert citation.similarity_score == 0.9
    prompt = llm.calls[0]
    assert "[S1]" in prompt["user_prompt"]
    assert "Glue retries use exponential backoff." in prompt["user_prompt"]
    assert "answer only from provided context" in prompt[
        "system_prompt"
    ].lower()


def test_retrieval_filters_scope_threshold_deduplicates_and_limits_context():
    valid = _result(text="0123456789ABCDEFGHIJ", score=0.9)
    duplicate = _result(text="duplicate", score=0.85)
    wrong_client = _result(
        document_id="other-client",
        chunk_id="other-client",
        client_id="client-b",
    )
    wrong_environment = _result(
        document_id="other-environment",
        chunk_id="other-environment",
        environment="prod",
    )
    below_threshold = _result(
        document_id="low",
        chunk_id="low",
        score=0.1,
    )
    config = ApplicationConfig(
        context_length_limit=10,
        maximum_retrieved_chunks=4,
        minimum_similarity=0.5,
    )
    retriever = StubRetriever(
        [
            wrong_client,
            wrong_environment,
            below_threshold,
            valid,
            duplicate,
        ]
    )
    llm = DeterministicFakeLLMProvider(response_text="Grounded [S1].")

    response = _application(
        retriever=retriever,
        llm_provider=llm,
        config=config,
    ).handle(_request("Find this in the runbook"))

    assert response.status == ApplicationStatus.COMPLETED
    assert response.retrieval_metadata.result_count == 1
    assert response.retrieval_metadata.filtered_for_scope == 2
    assert response.retrieval_metadata.deduplicated == 1
    assert response.retrieval_metadata.context_characters == 10
    assert retriever.calls[0]["top_k"] == 4
    assert retriever.calls[0]["minimum_similarity"] == 0.5
    assert "0123456789" in llm.calls[0]["user_prompt"]
    assert "ABCDEFGHIJ" not in llm.calls[0]["user_prompt"]


def test_empty_retrieval_returns_insufficient_context_without_llm_call():
    llm = DeterministicFakeLLMProvider()

    response = _application(
        retriever=StubRetriever(),
        llm_provider=llm,
    ).handle(_request("Find the recovery process in the runbook"))

    assert response.status == ApplicationStatus.INSUFFICIENT_CONTEXT
    assert "without guessing" in response.answer
    assert response.sources == ()
    assert response.retrieval_metadata.attempted is True
    assert llm.calls == []


def test_explicit_llm_insufficient_context_preserves_available_sources():
    llm = DeterministicFakeLLMProvider(
        response_text="",
        insufficient_context=True,
    )

    response = _application(
        retriever=StubRetriever([_result()]),
        llm_provider=llm,
    ).handle(_request("Find this in the runbook"))

    assert response.status == ApplicationStatus.INSUFFICIENT_CONTEXT
    assert len(response.sources) == 1
    assert response.model_metadata.finish_reason == "insufficient_context"
    assert "explicitly reported" in response.warnings[0]


def test_llm_insufficient_context_marker_is_a_typed_application_status():
    llm = DeterministicFakeLLMProvider(
        response_text=(
            "INSUFFICIENT_CONTEXT: The runbook does not contain this."
        ),
    )

    response = _application(
        retriever=StubRetriever([_result()]),
        llm_provider=llm,
    ).handle(_request("Find this in the runbook"))

    assert response.status == ApplicationStatus.INSUFFICIENT_CONTEXT
    assert "without guessing" in response.answer


def test_direct_response_uses_llm_without_retrieval():
    retriever = StubRetriever(error=AssertionError("must not retrieve"))
    llm = DeterministicFakeLLMProvider(response_text="Hello.")

    response = _application(
        retriever=retriever,
        llm_provider=llm,
    ).handle(_request("Hello"))

    assert response.status == ApplicationStatus.COMPLETED
    assert response.route == Route.DIRECT_RESPONSE
    assert response.sources == ()
    assert response.retrieval_metadata.attempted is False
    assert len(llm.calls) == 1
    assert retriever.calls == []


@pytest.mark.parametrize(
    ("query", "expected_route", "prompt_text"),
    [
        (
            "Define pipeline requirements",
            Route.REQUIREMENTS_GATHERING,
            "missing requirements",
        ),
        (
            "Write SQL query for daily totals",
            Route.CODE_GENERATION,
            "state assumptions",
        ),
    ],
)
def test_nonretrieval_route_specific_prompt_behavior(
    query,
    expected_route,
    prompt_text,
):
    llm = DeterministicFakeLLMProvider(response_text="Structured response.")

    response = _application(llm_provider=llm).handle(_request(query))

    assert response.status == ApplicationStatus.COMPLETED
    assert response.route == expected_route
    assert prompt_text in llm.calls[0]["system_prompt"].lower()


def test_troubleshooting_retrieves_and_builds_diagnostic_prompt():
    llm = DeterministicFakeLLMProvider(
        response_text="Check retry metrics [S1]."
    )

    response = _application(
        retriever=StubRetriever([_result()]),
        llm_provider=llm,
    ).handle(_request("My Glue job failed"))

    assert response.route == Route.TROUBLESHOOTING
    assert response.status == ApplicationStatus.COMPLETED
    assert "diagnostic steps" in llm.calls[0]["system_prompt"]


def test_deployment_requires_approval_and_invokes_no_providers():
    embedding = Mock()
    llm = Mock()
    retriever = StubRetriever(error=AssertionError("must not retrieve"))

    response = _application(
        retriever=retriever,
        embedding_provider=embedding,
        llm_provider=llm,
    ).handle(_request("Deploy CDK"))

    assert response.status == ApplicationStatus.APPROVAL_REQUIRED
    assert response.approval_required is True
    assert "No action was executed" in response.answer
    embedding.embed.assert_not_called()
    llm.generate.assert_not_called()
    assert retriever.calls == []


def test_destructive_request_requires_safety_review_and_no_execution():
    embedding = Mock()
    llm = Mock()

    response = _application(
        embedding_provider=embedding,
        llm_provider=llm,
    ).handle(_request("Delete the production bucket"))

    assert response.status == ApplicationStatus.SAFETY_REVIEW_REQUIRED
    assert response.approval_required is True
    assert response.safety_review_required is True
    assert response.route == Route.REJECTION_OR_SAFETY_REVIEW
    embedding.embed.assert_not_called()
    llm.generate.assert_not_called()


def test_tool_execution_route_does_not_execute_or_claim_current_state():
    llm = Mock()

    response = _application(llm_provider=llm).handle(
        _request("Check current alarm status")
    )

    assert response.route == Route.TOOL_EXECUTION
    assert response.status == ApplicationStatus.INSUFFICIENT_CONTEXT
    assert "no action was taken" in response.answer.lower()
    llm.generate.assert_not_called()


@pytest.mark.parametrize(
    ("overrides", "query", "category"),
    [
        (
            {"classifier": FailingClassifier()},
            "Hello",
            "classification_failure",
        ),
        (
            {"router": FailingRouter()},
            "Hello",
            "routing_failure",
        ),
        (
            {
                "embedding_provider": DeterministicFakeEmbeddingProvider(
                    fail_on_texts=frozenset({"Find this in the runbook"})
                )
            },
            "Find this in the runbook",
            "embedding_failure",
        ),
        (
            {
                "retriever": StubRetriever(
                    error=RuntimeError("private retrieval details")
                )
            },
            "Find this in the runbook",
            "retrieval_failure",
        ),
        (
            {"prompt_builder": FailingPromptBuilder()},
            "Hello",
            "prompt_construction_failure",
        ),
        (
            {
                "llm_provider": DeterministicFakeLLMProvider(
                    simulated_error=RuntimeError(
                        "private provider details"
                    )
                )
            },
            "Hello",
            "llm_invocation_failure",
        ),
    ],
)
def test_stage_failures_return_safe_typed_responses(
    overrides,
    query,
    category,
):
    response = _application(**overrides).handle(_request(query))

    assert response.status == ApplicationStatus.FAILED
    assert response.error_category == category
    assert "private" not in response.answer


@pytest.mark.parametrize(
    ("provider", "query", "category", "answer_fragment"),
    [
        (
            DeterministicFakeLLMProvider(
                simulated_error=LLMAccessDeniedError("private")
            ),
            "Hello",
            "provider_access_denied",
            "Model access was denied",
        ),
        (
            DeterministicFakeLLMProvider(
                simulated_error=LLMThrottledError("private")
            ),
            "Hello",
            "provider_throttled",
            "throttling requests",
        ),
        (
            DeterministicFakeLLMProvider(
                simulated_error=LLMModelUnavailableError("private")
            ),
            "Hello",
            "provider_unavailable",
            "model is currently unavailable",
        ),
    ],
)
def test_application_preserves_safe_provider_error_categories(
    provider,
    query,
    category,
    answer_fragment,
):
    response = _application(llm_provider=provider).handle(_request(query))

    assert response.status == ApplicationStatus.FAILED
    assert response.error_category == category
    assert answer_fragment in response.answer
    assert "private" not in response.answer


def test_embedding_provider_throttling_returns_safe_category():
    provider = DeterministicFakeEmbeddingProvider(
        fail_on_texts=frozenset({"Find this in the runbook"})
    )
    provider.embed = Mock(
        side_effect=EmbeddingThrottledError("private")
    )

    response = _application(
        embedding_provider=provider
    ).handle(_request("Find this in the runbook"))

    assert response.status == ApplicationStatus.FAILED
    assert response.error_category == "provider_throttled"
    assert "throttling requests" in response.answer


@pytest.mark.parametrize(
    "result",
    [
        {"generated_text": "not typed"},
        GenerationResult(
            generated_text="",
            model_id="invalid-empty",
        ),
    ],
)
def test_malformed_provider_results_fail_safely(result):
    response = _application(
        llm_provider=InvalidLLMProvider(result)
    ).handle(_request("Hello"))

    assert response.status == ApplicationStatus.FAILED
    assert response.error_category == "llm_invocation_failure"


def test_conversation_preserves_roles_and_truncates_without_persistence():
    config = ApplicationConfig(
        context_length_limit=9,
        maximum_conversation_messages=2,
    )
    llm = DeterministicFakeLLMProvider(response_text="Response.")
    conversation = (
        ConversationMessage(
            ConversationRole.USER,
            "old-message",
            "client-a",
            "dev",
        ),
        ConversationMessage(
            ConversationRole.ASSISTANT,
            "assistant",
            "client-a",
            "dev",
        ),
        ConversationMessage(
            ConversationRole.USER,
            "last",
            "client-a",
            "dev",
        ),
    )

    response = _application(
        llm_provider=llm,
        config=config,
    ).handle(
        _request("Hello", conversation_context=conversation)
    )

    prompt = llm.calls[0]["user_prompt"]
    assert "[USER] last" in prompt
    assert "old-message" not in prompt
    assert response.warnings == (
        "Prior conversation was truncated to configured limits.",
    )


@pytest.mark.parametrize(
    "message",
    [
        ConversationMessage(
            ConversationRole.USER,
            "other client",
            "client-b",
            "dev",
        ),
        ConversationMessage(
            ConversationRole.USER,
            "other environment",
            "client-a",
            "prod",
        ),
    ],
)
def test_application_request_rejects_mixed_conversation_scope(message):
    with pytest.raises(ValueError, match="cannot mix"):
        _request("Hello", conversation_context=(message,))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"request_id": " "}, "request_id"),
        ({"query": " "}, "query"),
        ({"client_id": "bad_client"}, "client_id"),
        ({"environment": "qa"}, "environment"),
    ],
)
def test_application_request_validates_required_fields(kwargs, message):
    values = {
        "query": "Hello",
        "request_id": "request-1",
        "client_id": "client-a",
        "environment": "dev",
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        _request(**values)


def test_configured_query_limit_returns_safe_failure():
    response = _application(
        config=ApplicationConfig(query_length_limit=4)
    ).handle(_request("Hello"))

    assert response.status == ApplicationStatus.FAILED
    assert response.error_category == "application_failure"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bedrock_llm_region": "invalid"}, "bedrock_llm_region"),
        ({"bedrock_llm_model_id": " "}, "bedrock_llm_model_id"),
        ({"temperature": 1.1}, "temperature"),
        ({"maximum_tokens": 0}, "maximum_tokens"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"query_length_limit": 0}, "query_length_limit"),
        ({"context_length_limit": 0}, "context_length_limit"),
        (
            {"maximum_conversation_messages": -1},
            "maximum_conversation_messages",
        ),
        (
            {"maximum_retrieved_chunks": 0},
            "maximum_retrieved_chunks",
        ),
        ({"minimum_similarity": 1.1}, "minimum_similarity"),
        ({"prompt_version": " "}, "prompt_version"),
        ({"application_version": " "}, "application_version"),
    ],
)
def test_application_config_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ApplicationConfig(**kwargs)


def test_application_requires_matching_prompt_version():
    with pytest.raises(ValueError, match="version"):
        _application(
            prompt_builder=GroundedPromptBuilder(
                prompt_version="different-v1"
            )
        )


def test_structured_logging_excludes_sensitive_content(caplog):
    logger = logging.getLogger("test.rag.application")
    caplog.set_level(logging.INFO, logger=logger.name)
    secret = "credential-secret-value"

    response = _application(
        llm_provider=DeterministicFakeLLMProvider(
            response_text="Safe response."
        ),
        event_logger=logger,
    ).handle(
        _request(
            f"Hello {secret}",
            metadata={"sensitive": True},
        )
    )

    assert response.status == ApplicationStatus.COMPLETED
    event = json.loads(caplog.records[-1].message)
    assert event["request_id"] == "request-1"
    assert event["client_id"] == "client-a"
    assert event["environment"] == "dev"
    assert event["intent"] == "general_conversation"
    assert event["route"] == "direct_response"
    assert event["retrieval_result_count"] == 0
    assert event["model_id"] == "fake-llm-v1"
    assert event["status"] == "completed"
    assert event["elapsed_ms"] >= 0
    assert secret not in caplog.text
    assert "system_prompt" not in caplog.text
    assert "user_prompt" not in caplog.text


def test_structured_logging_includes_non_sensitive_cost_metadata(caplog):
    logger = logging.getLogger("test.rag.cost")
    caplog.set_level(logging.INFO, logger=logger.name)

    response = _application(
        event_logger=logger,
        cost_estimator=CatalogCostEstimator([]),
        runtime_mode="demo",
    ).handle(_request("Hello"))

    event = json.loads(caplog.records[-1].message)
    estimate = response.model_metadata.cost_estimate
    assert estimate is not None
    assert event["model_id"] == "fake-llm-v1"
    assert event["input_token_count"] is not None
    assert event["output_token_count"] is not None
    assert event["pricing_version"] == "demo-no-charge-v1"
    assert event["estimated_total_cost"] == "0"
    assert event["currency"] == "USD"


def test_fake_llm_is_deterministic_configurable_and_preserves_metadata():
    provider = DeterministicFakeLLMProvider(
        response_text="default",
        responses_by_query={"special": "selected"},
        model_id="fake-custom",
        provider_metadata={"fixture": "stable"},
    )

    first = provider.generate(
        system_prompt="system",
        user_prompt="a special request",
        model_parameters={"temperature": 0},
    )
    second = provider.generate(
        system_prompt="system",
        user_prompt="a special request",
        model_parameters={"temperature": 0},
    )

    assert isinstance(provider, LLMProvider)
    assert first == second
    assert first.generated_text == "selected"
    assert first.model_id == "fake-custom"
    assert first.provider_metadata == {"fixture": "stable"}


def test_fake_llm_supports_simulated_errors():
    provider = DeterministicFakeLLMProvider(
        simulated_error=LLMInvocationError("simulated")
    )

    with pytest.raises(LLMInvocationError, match="simulated"):
        provider.generate(system_prompt="system", user_prompt="user")


def test_bedrock_converse_request_and_response_parsing():
    client = Mock()
    client.converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {"text": "First."},
                    {"text": "Second."},
                ]
            }
        },
        "usage": {"inputTokens": 12, "outputTokens": 4},
        "stopReason": "end_turn",
        "ResponseMetadata": {"RequestId": "aws-request"},
    }
    provider = BedrockLLMProvider(
        model_id="anthropic.test",
        region_name="us-west-2",
        temperature=0.2,
        maximum_tokens=300,
        timeout_seconds=10,
        bedrock_runtime_client=client,
    )

    result = provider.generate(
        system_prompt="Safe system prompt",
        user_prompt="Safe user prompt",
    )

    assert result.generated_text == "First.\nSecond."
    assert result.model_id == "anthropic.test"
    assert result.input_token_count == 12
    assert result.output_token_count == 4
    assert result.finish_reason == "end_turn"
    assert result.provider_metadata == {
        "provider": "amazon-bedrock",
        "request_id": "aws-request",
    }
    assert client.converse.call_args.kwargs == {
        "modelId": "anthropic.test",
        "system": [{"text": "Safe system prompt"}],
        "messages": [
            {
                "role": "user",
                "content": [{"text": "Safe user prompt"}],
            }
        ],
        "inferenceConfig": {
            "temperature": 0.2,
            "maxTokens": 300,
        },
    }


@pytest.mark.parametrize(
    ("code", "expected_error"),
    [
        ("ThrottlingException", LLMThrottledError),
        ("AccessDeniedException", LLMAccessDeniedError),
        ("ResourceNotFoundException", LLMModelUnavailableError),
        ("ModelNotReadyException", LLMModelUnavailableError),
        ("ServiceUnavailableException", LLMModelUnavailableError),
        ("ValidationException", LLMValidationError),
        ("InternalServerException", LLMInvocationError),
    ],
)
def test_bedrock_llm_translates_service_errors(code, expected_error):
    client = Mock()
    client.converse.side_effect = FakeServiceError(code)
    provider = BedrockLLMProvider(
        model_id="model-id",
        region_name="us-west-2",
        bedrock_runtime_client=client,
    )

    with pytest.raises(expected_error):
        provider.generate(system_prompt="system", user_prompt="user")


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {"output": {"message": {"content": []}}},
        {"output": {"message": {"content": [{"not_text": "x"}]}}},
        {
            "output": {"message": {"content": [{"text": "answer"}]}},
            "usage": {"inputTokens": -1},
        },
        {
            "output": {"message": {"content": [{"text": "answer"}]}},
            "stopReason": 5,
        },
    ],
)
def test_bedrock_llm_rejects_malformed_responses(response):
    client = Mock()
    client.converse.return_value = response
    provider = BedrockLLMProvider(
        model_id="model-id",
        region_name="us-west-2",
        bedrock_runtime_client=client,
    )

    with pytest.raises(MalformedLLMResponseError):
        provider.generate(system_prompt="system", user_prompt="user")


@pytest.mark.parametrize(
    "parameters",
    [
        {"unknown": True},
        {"temperature": 2},
        {"maximum_tokens": 0},
    ],
)
def test_bedrock_llm_validates_model_parameters_without_invocation(
    parameters,
):
    client = Mock()
    provider = BedrockLLMProvider(
        model_id="model-id",
        region_name="us-west-2",
        bedrock_runtime_client=client,
    )

    with pytest.raises(LLMValidationError):
        provider.generate(
            system_prompt="system",
            user_prompt="user",
            model_parameters=parameters,
        )
    client.converse.assert_not_called()


def test_prompt_builder_contains_required_grounding_and_safety_instructions():
    llm = DeterministicFakeLLMProvider(response_text="Answer [S1].")

    _application(
        retriever=StubRetriever([_result()]),
        llm_provider=llm,
    ).handle(_request("Find this in the runbook"))

    system = llm.calls[0]["system_prompt"].lower()
    user = llm.calls[0]["user_prompt"]
    for required in (
        "provided context",
        "context is insufficient",
        "do not invent aws resources",
        "cite",
        "recommendations from confirmed facts",
        "never claim an action was executed",
        "never authorization",
    ):
        assert required in system
    assert "client_id: client-a" in user
    assert "environment: dev" in user
    assert "source=operations-runbook" in user


def test_rag_evaluation_computes_deterministic_checks():
    app = _application(
        retriever=StubRetriever([_result()]),
        llm_provider=DeterministicFakeLLMProvider(
            response_text="Use exponential backoff [S1]."
        ),
    )
    case = RAGEvaluationCase(
        query="Find retry guidance in the runbook",
        expected_intent=Intent.KNOWLEDGE_QUESTION,
        expected_source_ids=frozenset({"S1"}),
        reference_answer="Retries should use exponential backoff.",
        required_facts=("exponential backoff",),
        forbidden_claims=("deployment completed",),
        client_id="client-a",
        environment="dev",
    )

    summary = RAGEvaluator().evaluate(app, (case,))

    assert summary.intent_accuracy == 1.0
    assert summary.source_recall == 1.0
    assert summary.required_fact_rate == 1.0
    assert summary.forbidden_claim_avoidance_rate == 1.0
    assert summary.insufficient_context_accuracy == 1.0


def test_rag_evaluation_checks_insufficient_context_correctness():
    app = _application(retriever=StubRetriever())
    case = RAGEvaluationCase(
        query="Find missing information in the runbook",
        expected_intent=Intent.KNOWLEDGE_QUESTION,
        expect_insufficient_context=True,
        client_id="client-a",
        environment="dev",
    )

    summary = RAGEvaluator().evaluate(app, (case,))

    assert summary.insufficient_context_accuracy == 1.0
