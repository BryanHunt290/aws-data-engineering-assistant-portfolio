"""Provider-neutral embedding error taxonomy."""


class EmbeddingProviderError(RuntimeError):
    """Base class for embedding provider failures."""


class EmbeddingThrottledError(EmbeddingProviderError):
    """The provider rejected a request because a quota was exceeded."""


class EmbeddingAccessDeniedError(EmbeddingProviderError):
    """The caller is not authorized to invoke the embedding model."""


class EmbeddingModelUnavailableError(EmbeddingProviderError):
    """The selected model is missing, not ready, or temporarily unavailable."""


class MalformedEmbeddingResponseError(EmbeddingProviderError):
    """The provider returned a response that cannot be used safely."""


class EmbeddingInvocationError(EmbeddingProviderError):
    """The provider failed for an unclassified reason."""


class OllamaEmbeddingUnavailableError(EmbeddingModelUnavailableError):
    """The configured local Ollama service or model is unavailable."""


class OllamaEmbeddingTimeoutError(OllamaEmbeddingUnavailableError):
    """The local Ollama embedding request exceeded its timeout."""
