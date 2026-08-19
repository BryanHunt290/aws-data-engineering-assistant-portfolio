from unittest.mock import Mock

import pytest
import requests

from knowledge.llm import LLMProvider
from knowledge.llm_errors import (
    LLMInvocationError,
    LLMThrottledError,
    LLMValidationError,
    MalformedLLMResponseError,
)
from knowledge.ollama_llm import (
    OllamaLLMProvider,
    OllamaTimeoutError,
    OllamaUnavailableError,
)


def _response(payload, *, status_code=200):
    response = Mock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


def test_ollama_provider_builds_bounded_local_chat_request():
    session = Mock()
    session.post.return_value = _response(
        {
            "model": "gpt-oss:20b",
            "message": {"role": "assistant", "content": " Local answer "},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 17,
            "eval_count": 4,
        }
    )
    provider = OllamaLLMProvider(
        http_session=session,
        connect_timeout_seconds=3,
        read_timeout_seconds=90,
    )

    result = provider.generate(
        system_prompt=" System ",
        user_prompt=" User ",
        model_parameters={
            "temperature": 0,
            "maximum_tokens": 300,
            "response_format": {
                "type": "object",
                "properties": {},
            },
        },
    )

    assert isinstance(provider, LLMProvider)
    assert result.generated_text == "Local answer"
    assert result.model_id == "gpt-oss:20b"
    assert result.input_token_count == 17
    assert result.output_token_count == 4
    assert result.provider_metadata == {"provider": "ollama"}
    call = session.post.call_args
    assert call.args == ("http://localhost:11434/api/chat",)
    assert call.kwargs["timeout"] == (3.0, 90.0)
    payload = call.kwargs["json"]
    assert payload["model"] == "gpt-oss:20b"
    assert payload["messages"] == [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "User"},
    ]
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"] == {
        "temperature": 0.0,
        "num_predict": 300,
    }
    assert payload["format"]["type"] == "object"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (requests.Timeout("private"), OllamaTimeoutError),
        (requests.ConnectionError("private"), OllamaUnavailableError),
        (requests.RequestException("private"), LLMInvocationError),
    ],
)
def test_ollama_network_failures_are_typed_and_non_sensitive(error, expected):
    session = Mock()
    session.post.side_effect = error
    provider = OllamaLLMProvider(http_session=session)

    with pytest.raises(expected) as raised:
        provider.generate(system_prompt="system", user_prompt="user")

    assert "private" not in str(raised.value)


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (429, LLMThrottledError),
        (404, OllamaUnavailableError),
        (400, LLMValidationError),
        (422, LLMValidationError),
        (500, LLMInvocationError),
    ],
)
def test_ollama_http_failures_are_translated(status_code, expected):
    session = Mock()
    session.post.return_value = _response({}, status_code=status_code)
    provider = OllamaLLMProvider(http_session=session)

    with pytest.raises(expected):
        provider.generate(system_prompt="system", user_prompt="user")


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"message": {"content": ""}, "done": True},
        {"message": {"content": "partial"}, "done": False},
        {
            "message": {"content": "answer"},
            "done": True,
            "prompt_eval_count": -1,
        },
    ],
)
def test_ollama_malformed_responses_are_rejected(payload):
    session = Mock()
    session.post.return_value = _response(payload)
    provider = OllamaLLMProvider(http_session=session)

    with pytest.raises(MalformedLLMResponseError):
        provider.generate(system_prompt="system", user_prompt="user")


def test_ollama_invalid_json_is_rejected():
    session = Mock()
    response = _response({})
    response.json.side_effect = requests.JSONDecodeError(
        "invalid",
        "<html>",
        0,
    )
    session.post.return_value = response
    provider = OllamaLLMProvider(http_session=session)

    with pytest.raises(MalformedLLMResponseError):
        provider.generate(system_prompt="system", user_prompt="user")


@pytest.mark.parametrize(
    "base_url",
    [
        "https://example.com:11434",
        "http://user:pass@localhost:11434",
        "http://localhost:11434/path",
        "http://localhost",
        "file://localhost:11434",
    ],
)
def test_ollama_rejects_non_loopback_or_ambiguous_urls(base_url):
    with pytest.raises(ValueError):
        OllamaLLMProvider(base_url=base_url, http_session=Mock())


def test_ollama_rejects_unknown_model_parameters_without_network_call():
    session = Mock()
    provider = OllamaLLMProvider(http_session=session)

    with pytest.raises(LLMValidationError):
        provider.generate(
            system_prompt="system",
            user_prompt="user",
            model_parameters={"secret": "not-supported"},
        )

    session.post.assert_not_called()
