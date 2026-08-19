from unittest.mock import Mock

import pytest
import requests

from knowledge.embedding_errors import (
    EmbeddingInvocationError,
    EmbeddingThrottledError,
    MalformedEmbeddingResponseError,
    OllamaEmbeddingTimeoutError,
    OllamaEmbeddingUnavailableError,
)
from knowledge.embeddings import EmbeddingProvider
from knowledge.ollama_embeddings import OllamaEmbeddingProvider


def _response(payload, *, status_code=200):
    response = Mock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


def test_ollama_embedding_provider_sends_configured_model_and_batch():
    session = Mock()
    session.post.return_value = _response(
        {
            "model": "embeddinggemma:latest",
            "embeddings": [[1.0, 0.0], [0.0, 1.0]],
        }
    )
    provider = OllamaEmbeddingProvider(
        base_url="http://127.0.0.1:11434/",
        model_id="embeddinggemma:latest",
        connect_timeout_seconds=2,
        read_timeout_seconds=30,
        http_session=session,
    )

    vectors = provider.embed([" first ", "second"])

    assert isinstance(provider, EmbeddingProvider)
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    session.post.assert_called_once_with(
        "http://127.0.0.1:11434/api/embed",
        json={
            "model": "embeddinggemma:latest",
            "input": ["first", "second"],
        },
        timeout=(2.0, 30.0),
    )


def test_ollama_embedding_provider_supports_single_text():
    session = Mock()
    session.post.return_value = _response(
        {"model": "embeddinggemma", "embeddings": [[0.5, 0.5]]}
    )

    vector = OllamaEmbeddingProvider(http_session=session).embed_text("one")

    assert vector == [0.5, 0.5]


@pytest.mark.parametrize("texts", [[], [""], ["  "], ["valid", ""]])
def test_ollama_embedding_provider_rejects_empty_input(texts):
    session = Mock()
    provider = OllamaEmbeddingProvider(http_session=session)

    with pytest.raises(ValueError, match="empty"):
        provider.embed(texts)

    session.post.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"model": "embeddinggemma", "embeddings": []},
        {"model": "embeddinggemma", "embeddings": [[True]]},
        {"model": "embeddinggemma", "embeddings": [[float("inf")]]},
        {"model": "embeddinggemma", "embeddings": [[1.0], [2.0]]},
        {"model": "embeddinggemma", "embeddings": [[1.0], [1.0, 2.0]]},
    ],
)
def test_ollama_embedding_provider_rejects_malformed_response(payload):
    session = Mock()
    session.post.return_value = _response(payload)
    provider = OllamaEmbeddingProvider(http_session=session)

    with pytest.raises(MalformedEmbeddingResponseError):
        provider.embed(["input"])


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (requests.Timeout("private text"), OllamaEmbeddingTimeoutError),
        (
            requests.ConnectionError("private text"),
            OllamaEmbeddingUnavailableError,
        ),
        (requests.RequestException("private text"), EmbeddingInvocationError),
    ],
)
def test_ollama_embedding_network_errors_are_safe_and_typed(error, expected):
    session = Mock()
    session.post.side_effect = error

    with pytest.raises(expected) as raised:
        OllamaEmbeddingProvider(http_session=session).embed(["sensitive"])

    assert "private text" not in str(raised.value)
    assert "sensitive" not in str(raised.value)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (404, OllamaEmbeddingUnavailableError),
        (429, EmbeddingThrottledError),
        (500, EmbeddingInvocationError),
    ],
)
def test_ollama_embedding_http_errors_are_converted(status, expected):
    session = Mock()
    session.post.return_value = _response({}, status_code=status)

    with pytest.raises(expected):
        OllamaEmbeddingProvider(http_session=session).embed(["input"])


def test_ollama_embedding_construction_is_lazy():
    session = Mock()

    provider = OllamaEmbeddingProvider(
        model_id="custom-embedding-model",
        http_session=session,
    )

    assert provider.model_id == "custom-embedding-model"
    session.post.assert_not_called()
    session.get.assert_not_called()
