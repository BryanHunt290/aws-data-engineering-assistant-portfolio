"""Pure session-state helpers used by Streamlit and unit tests."""

from dataclasses import asdict, dataclass
import csv
from datetime import datetime, timezone
from io import StringIO
import json
from decimal import Decimal
from typing import Any, MutableMapping

from knowledge.application_models import (
    ApplicationResponse,
    ConversationMessage,
    ConversationRole,
)


HISTORY_KEY = "conversation_history"
FEEDBACK_KEY = "feedback_by_request"
SCOPE_KEY = "conversation_scope"
LAST_RESPONSE_KEY = "last_response"
USAGE_KEY = "cost_usage_by_request"


@dataclass(frozen=True)
class SessionMessage:
    """One in-memory display and conversation-context message."""

    role: ConversationRole
    content: str
    client_id: str
    environment: str
    request_id: str | None = None


@dataclass(frozen=True)
class FeedbackRecord:
    """One session-only response rating."""

    request_id: str
    rating: str
    comment: str
    created_at: str
    model_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_input_cost: str | None = None
    estimated_output_cost: str | None = None
    estimated_total_cost: str | None = None
    pricing_version: str | None = None


@dataclass(frozen=True)
class SessionCostTotals:
    """In-memory request/token totals with chargeable estimated cost."""

    request_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_estimated_cost: Decimal


def initialize_session(state: MutableMapping[str, Any]) -> None:
    """Create local state containers without persistence."""

    state.setdefault(HISTORY_KEY, [])
    state.setdefault(FEEDBACK_KEY, {})
    state.setdefault(SCOPE_KEY, None)
    state.setdefault(LAST_RESPONSE_KEY, None)
    state.setdefault(USAGE_KEY, {})


def ensure_scope(
    state: MutableMapping[str, Any],
    client_id: str,
    environment: str,
) -> bool:
    """Reset conversation when its client/environment boundary changes."""

    initialize_session(state)
    scope = (client_id, environment)
    if state[SCOPE_KEY] == scope:
        return False
    state[SCOPE_KEY] = scope
    state[HISTORY_KEY] = []
    state[LAST_RESPONSE_KEY] = None
    return True


def clear_conversation(state: MutableMapping[str, Any]) -> None:
    """Clear request-local conversation without deleting feedback."""

    initialize_session(state)
    state[HISTORY_KEY] = []
    state[LAST_RESPONSE_KEY] = None


def append_message(
    state: MutableMapping[str, Any],
    message: SessionMessage,
    *,
    maximum_messages: int,
) -> None:
    """Append and bound session-only history."""

    if maximum_messages < 0:
        raise ValueError("maximum_messages cannot be negative")
    initialize_session(state)
    current_scope = state[SCOPE_KEY]
    if current_scope != (message.client_id, message.environment):
        raise ValueError("Message scope does not match session scope")
    history = list(state[HISTORY_KEY])
    history.append(message)
    state[HISTORY_KEY] = (
        history[-maximum_messages:] if maximum_messages else []
    )


def conversation_context(
    state: MutableMapping[str, Any],
    *,
    client_id: str,
    environment: str,
    maximum_messages: int,
) -> tuple[ConversationMessage, ...]:
    """Return bounded backend context from exactly one scope."""

    if maximum_messages < 0:
        raise ValueError("maximum_messages cannot be negative")
    initialize_session(state)
    history = [
        message
        for message in state[HISTORY_KEY]
        if message.client_id == client_id
        and message.environment == environment
    ]
    if maximum_messages == 0:
        return ()
    return tuple(
        ConversationMessage(
            role=message.role,
            content=message.content,
            client_id=client_id,
            environment=environment,
        )
        for message in history[-maximum_messages:]
    )


def record_feedback(
    state: MutableMapping[str, Any],
    *,
    request_id: str,
    rating: str,
    comment: str = "",
    created_at: datetime | None = None,
    response: ApplicationResponse | None = None,
) -> bool:
    """Record one rating per request and report whether it was accepted."""

    initialize_session(state)
    normalized_rating = rating.strip().lower()
    if normalized_rating not in {"up", "down"}:
        raise ValueError("rating must be up or down")
    if request_id in state[FEEDBACK_KEY]:
        return False
    timestamp = created_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("created_at must include timezone information")
    feedback = dict(state[FEEDBACK_KEY])
    model = response.model_metadata if response is not None else None
    estimate = model.cost_estimate if model is not None else None
    feedback[request_id] = FeedbackRecord(
        request_id=request_id,
        rating=normalized_rating,
        comment=comment.strip()[:500],
        created_at=timestamp.isoformat().replace("+00:00", "Z"),
        model_id=model.model_id if model is not None else None,
        input_tokens=(
            model.input_token_count if model is not None else None
        ),
        output_tokens=(
            model.output_token_count if model is not None else None
        ),
        estimated_input_cost=_decimal_text(
            estimate.input_cost if estimate is not None else None
        ),
        estimated_output_cost=_decimal_text(
            estimate.output_cost if estimate is not None else None
        ),
        estimated_total_cost=_decimal_text(
            estimate.total_estimated_cost
            if estimate is not None
            else None
        ),
        pricing_version=(
            estimate.pricing_version if estimate is not None else None
        ),
    )
    state[FEEDBACK_KEY] = feedback
    return True


def feedback_json(state: MutableMapping[str, Any]) -> str:
    """Export current-session feedback as deterministic JSON."""

    records = _feedback_records(state)
    return json.dumps(
        [asdict(record) for record in records],
        indent=2,
        sort_keys=True,
    )


def feedback_csv(state: MutableMapping[str, Any]) -> str:
    """Export current-session feedback as CSV."""

    records = _feedback_records(state)
    output = StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "request_id",
            "rating",
            "comment",
            "created_at",
            "model_id",
            "input_tokens",
            "output_tokens",
            "estimated_input_cost",
            "estimated_output_cost",
            "estimated_total_cost",
            "pricing_version",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(asdict(record) for record in records)
    return output.getvalue()


def _feedback_records(
    state: MutableMapping[str, Any],
) -> list[FeedbackRecord]:
    initialize_session(state)
    return [
        state[FEEDBACK_KEY][request_id]
        for request_id in sorted(state[FEEDBACK_KEY])
    ]


def accumulate_response_cost(
    state: MutableMapping[str, Any],
    response: ApplicationResponse,
) -> bool:
    """Accumulate one response exactly once for the browser session."""

    initialize_session(state)
    records = dict(state[USAGE_KEY])
    if response.request_id in records:
        return False
    model = response.model_metadata
    estimate = model.cost_estimate
    charge = Decimal("0")
    if (
        estimate is not None
        and estimate.is_chargeable
        and estimate.total_estimated_cost is not None
    ):
        charge = estimate.total_estimated_cost
    records[response.request_id] = {
        "input_tokens": model.input_token_count or 0,
        "output_tokens": model.output_token_count or 0,
        "estimated_cost": str(charge),
    }
    state[USAGE_KEY] = records
    return True


def session_cost_totals(
    state: MutableMapping[str, Any],
) -> SessionCostTotals:
    """Return current browser-session totals without persistence."""

    initialize_session(state)
    records = tuple(state[USAGE_KEY].values())
    return SessionCostTotals(
        request_count=len(records),
        total_input_tokens=sum(
            int(record["input_tokens"]) for record in records
        ),
        total_output_tokens=sum(
            int(record["output_tokens"]) for record in records
        ),
        total_estimated_cost=sum(
            (
                Decimal(record["estimated_cost"])
                for record in records
            ),
            start=Decimal("0"),
        ),
    )


def _decimal_text(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None
