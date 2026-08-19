"""Typed stage errors raised or translated by application orchestration."""


class ApplicationError(RuntimeError):
    """Base error with a safe category and user-facing message."""

    category = "application_failure"
    user_message = "The request could not be completed safely."


class ClassificationFailure(ApplicationError):
    category = "classification_failure"
    user_message = "The request could not be classified safely."


class RoutingFailure(ApplicationError):
    category = "routing_failure"
    user_message = "A safe handling route could not be selected."


class QueryEmbeddingFailure(ApplicationError):
    category = "embedding_failure"
    user_message = "The request could not be prepared for retrieval."


class RetrievalFailure(ApplicationError):
    category = "retrieval_failure"
    user_message = "Scoped knowledge retrieval failed."


class PromptConstructionFailure(ApplicationError):
    category = "prompt_construction_failure"
    user_message = "A safe model prompt could not be constructed."


class LLMInvocationFailure(ApplicationError):
    category = "llm_invocation_failure"
    user_message = "Response generation failed."


class InvalidScopeError(ApplicationError):
    category = "invalid_scope"
    user_message = "The request contains an invalid client or environment scope."


class InsufficientContextError(ApplicationError):
    category = "insufficient_context"
    user_message = "The available scoped knowledge is insufficient."


class ProviderThrottledFailure(ApplicationError):
    category = "provider_throttled"
    user_message = (
        "The model provider is throttling requests. Wait briefly and retry."
    )


class ProviderAccessDeniedFailure(ApplicationError):
    category = "provider_access_denied"
    user_message = (
        "Model access was denied. Check credentials, Region, model access, "
        "and IAM permissions."
    )


class ProviderUnavailableFailure(ApplicationError):
    category = "provider_unavailable"
    user_message = "The configured model is currently unavailable."


class MalformedProviderFailure(ApplicationError):
    category = "malformed_provider_response"
    user_message = (
        "The model provider returned an unusable response. No answer was shown."
    )
