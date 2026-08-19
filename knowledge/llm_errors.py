"""Provider-neutral errors for language-model generation."""


class LLMProviderError(RuntimeError):
    """Base class for language-model provider failures."""


class LLMThrottledError(LLMProviderError):
    """The provider rejected a request because of throttling."""


class LLMAccessDeniedError(LLMProviderError):
    """The caller cannot access the configured language model."""


class LLMModelUnavailableError(LLMProviderError):
    """The configured language model is unavailable."""


class LLMValidationError(LLMProviderError):
    """The provider rejected invalid generation parameters."""


class MalformedLLMResponseError(LLMProviderError):
    """The provider response did not satisfy the generation contract."""


class LLMInvocationError(LLMProviderError):
    """The provider failed for an unclassified reason."""
