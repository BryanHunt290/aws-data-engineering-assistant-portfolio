"""Safe presentation formatting independent of Streamlit."""

from typing import Any, Mapping

from knowledge.application_models import (
    ApplicationResponse,
    ApplicationStatus,
    SourceCitation,
)
from knowledge.embedding_errors import (
    EmbeddingAccessDeniedError,
    EmbeddingModelUnavailableError,
    EmbeddingThrottledError,
    MalformedEmbeddingResponseError,
)
from knowledge.llm_errors import (
    LLMAccessDeniedError,
    LLMModelUnavailableError,
    LLMThrottledError,
    MalformedLLMResponseError,
)
from knowledge.ollama_llm import (
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from knowledge.costs import CostEstimate


_DISPLAY_METADATA_KEYS = (
    "client_id",
    "environment",
    "topic",
    "file_type",
    "synthetic",
    "license",
)


def response_details(response: ApplicationResponse) -> dict[str, str]:
    """Return stable labels for the response detail panel."""

    model = response.model_metadata
    return {
        "Status": response.status.value,
        "Intent": response.intent.value,
        "Route": response.route.value,
        "Classifier confidence": f"{response.confidence:.1%}",
        "Approval required": _yes_no(response.approval_required),
        "Safety review required": _yes_no(
            response.safety_review_required
        ),
        "Total latency": f"{response.latency_ms:.1f} ms",
        "Model ID": model.model_id or "Not invoked",
        "Input tokens": _optional_number(model.input_token_count),
        "Output tokens": _optional_number(model.output_token_count),
        "Estimated cost": (
            model.cost_estimate.formatted_total
            if model.cost_estimate is not None
            else "Unavailable"
        ),
    }


def cost_details(estimate: CostEstimate) -> dict[str, str]:
    """Return safe, presentation-ready component details."""

    return {
        "Model ID": estimate.model_id,
        "Input tokens": _optional_number(estimate.input_token_count),
        "Input token rate": _rate(
            estimate.input_price_per_million_tokens,
            estimate.currency,
        ),
        "Input estimated cost": _cost(
            estimate.input_cost, estimate.currency
        ),
        "Output tokens": _optional_number(estimate.output_token_count),
        "Output token rate": _rate(
            estimate.output_price_per_million_tokens,
            estimate.currency,
        ),
        "Output estimated cost": _cost(
            estimate.output_cost, estimate.currency
        ),
        "Cache-read tokens": _optional_number(
            estimate.cache_read_token_count
        ),
        "Cache-read rate": _rate(
            estimate.cache_read_price_per_million_tokens,
            estimate.currency,
        ),
        "Cache-read estimated cost": _cost(
            estimate.cache_read_cost, estimate.currency
        ),
        "Cache-write tokens": _optional_number(
            estimate.cache_write_token_count
        ),
        "Cache-write rate": _rate(
            estimate.cache_write_price_per_million_tokens,
            estimate.currency,
        ),
        "Cache-write estimated cost": _cost(
            estimate.cache_write_cost, estimate.currency
        ),
        "Total estimated cost": estimate.formatted_total,
        "Pricing source": estimate.pricing_source or "Not available",
        "Effective date": (
            estimate.pricing_effective_date or "Not available"
        ),
        "Pricing version": estimate.pricing_version or "Not available",
        "Warning": estimate.estimate_warning or "None",
    }


def status_presentation(
    status: ApplicationStatus,
) -> tuple[str, str]:
    """Return a Streamlit-neutral severity and heading."""

    return {
        ApplicationStatus.COMPLETED: ("success", "Completed"),
        ApplicationStatus.APPROVAL_REQUIRED: (
            "warning",
            "Explicit approval required",
        ),
        ApplicationStatus.SAFETY_REVIEW_REQUIRED: (
            "error",
            "Safety review required",
        ),
        ApplicationStatus.INSUFFICIENT_CONTEXT: (
            "warning",
            "Insufficient context",
        ),
        ApplicationStatus.FAILED: ("error", "Request failed"),
    }[status]


def source_details(source: SourceCitation) -> dict[str, Any]:
    """Return attributed fields plus an allowlist of source metadata."""

    selected_metadata = {
        key: source.metadata[key]
        for key in _DISPLAY_METADATA_KEYS
        if key in source.metadata
    }
    return {
        "source_id": source.source_id,
        "source_name": source.source_name,
        "document_id": source.document_id,
        "chunk_id": source.chunk_id,
        "similarity_score": round(source.similarity_score, 4),
        "page": source.page,
        "section": source.section,
        "object_key": source.object_key,
        "metadata": selected_metadata,
    }


def source_summary(source: SourceCitation) -> str:
    """Create a context-free developer summary with no document text."""

    location = []
    if source.page is not None:
        location.append(f"page={source.page}")
    if source.section is not None:
        location.append(f"section={source.section}")
    suffix = f" ({', '.join(location)})" if location else ""
    return (
        f"[{source.source_id}] {source.source_name} / "
        f"{source.document_id} / {source.chunk_id}; "
        f"score={source.similarity_score:.4f}{suffix}"
    )


def safe_error_message(error: Exception) -> str:
    """Map runtime/bootstrap failures to non-sensitive UI text."""

    if isinstance(error, OllamaTimeoutError):
        return (
            "The local Ollama request timed out. Confirm the model is running "
            "and retry."
        )
    if isinstance(error, OllamaUnavailableError):
        return (
            "Local Ollama or the configured model is unavailable. Start "
            "Ollama, verify the model, and retry."
        )
    if isinstance(
        error,
        (LLMThrottledError, EmbeddingThrottledError),
    ):
        return (
            "Amazon Bedrock is throttling requests. Wait briefly and retry."
        )
    if isinstance(
        error,
        (LLMAccessDeniedError, EmbeddingAccessDeniedError),
    ):
        return (
            "Bedrock access was denied. Check credentials, model access, "
            "Region, and IAM permissions."
        )
    if isinstance(
        error,
        (LLMModelUnavailableError, EmbeddingModelUnavailableError),
    ):
        return (
            "The configured Bedrock model is unavailable in this Region."
        )
    if isinstance(
        error,
        (MalformedLLMResponseError, MalformedEmbeddingResponseError),
    ):
        return (
            "The model returned an unusable response. No answer was shown."
        )
    if type(error).__name__ in {
        "NoCredentialsError",
        "PartialCredentialsError",
    }:
        return (
            "AWS credentials were not found or are incomplete. Configure the "
            "standard AWS credential chain outside the UI."
        )
    if isinstance(error, ValueError):
        return f"Configuration error: {error}"
    return (
        "The request could not be completed safely. Check local configuration "
        "and try again."
    )


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _optional_number(value: int | None) -> str:
    return str(value) if value is not None else "Not available"


def _cost(value, currency: str | None) -> str:
    if value is None:
        return "Unavailable"
    prefix = "$" if currency == "USD" else f"{currency} "
    return f"{prefix}{value:.6f}"


def _rate(value, currency: str | None) -> str:
    if value is None:
        return "Not available"
    prefix = "$" if currency == "USD" else f"{currency} "
    return f"{prefix}{value} per 1M tokens"
