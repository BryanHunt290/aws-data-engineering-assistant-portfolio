"""Provider-neutral monitoring events and an offline JSON Lines sink."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import StrEnum
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


CURRENT_MONITORING_SCHEMA_VERSION = 1
DEFAULT_MONITORING_PATH = Path("data/monitoring/events.jsonl")
MAXIMUM_RECORD_CHARACTERS = 100_000
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_SCOPE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "credential",
        "credentials",
        "document_text",
        "password",
        "prompt",
        "query",
        "raw_prompt",
        "secret",
        "session_token",
    }
)
_SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(
        r"(?i)\b(?:password|secret|token|authorization)\s*[:=]\s*\S+"
    ),
)


class MonitoringEventType(StrEnum):
    """Supported provider-neutral monitoring event categories."""

    APPLICATION_REQUEST = "application_request"
    RETRIEVAL_COMPLETION = "retrieval_completion"
    LLM_COMPLETION = "llm_completion"
    SAFETY_DECISION = "safety_decision"
    APPROVAL_REQUIREMENT = "approval_requirement"
    USER_FEEDBACK = "user_feedback"
    EVALUATION_RUN = "evaluation_run"
    APPLICATION_ERROR = "application_error"


class SafetyOutcome(StrEnum):
    """Safe, non-executing outcome recorded for routed requests."""

    ALLOWED = "allowed"
    BLOCKED = "blocked"
    APPROVAL_REQUIRED = "approval_required"
    SAFETY_REVIEW_REQUIRED = "safety_review_required"


class UserRating(StrEnum):
    """Ratings compatible with the existing Streamlit feedback model."""

    UP = "up"
    DOWN = "down"


@dataclass(frozen=True)
class MonitoringEvent:
    """One scoped, privacy-minimized monitoring record."""

    event_id: str
    event_type: MonitoringEventType
    timestamp: datetime
    client_id: str
    environment: str
    runtime_mode: str
    schema_version: int = CURRENT_MONITORING_SCHEMA_VERSION
    session_id: str | None = None
    request_id: str | None = None
    intent: str | None = None
    retrieval_strategy: str | None = None
    prompt_strategy: str | None = None
    llm_provider: str | None = None
    model_id: str | None = None
    response_mode: str | None = None
    latency_ms: float | None = None
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: Decimal | None = None
    retrieved_source_count: int | None = None
    citation_count: int | None = None
    approval_required: bool | None = None
    safety_outcome: SafetyOutcome | None = None
    success: bool | None = None
    error_category: str | None = None
    user_rating: UserRating | None = None
    feedback_text: str | None = None
    evaluation_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != CURRENT_MONITORING_SCHEMA_VERSION:
            raise ValueError("Unsupported monitoring schema_version")
        if not isinstance(self.event_id, str) or not _IDENTIFIER.fullmatch(
            self.event_id.strip()
        ):
            raise ValueError("event_id is invalid")
        try:
            event_type = MonitoringEventType(self.event_type)
        except (TypeError, ValueError) as error:
            raise ValueError("event_type is invalid") from error
        if (
            not isinstance(self.timestamp, datetime)
            or self.timestamp.tzinfo is None
            or self.timestamp.utcoffset() is None
        ):
            raise ValueError("timestamp must include timezone information")
        timestamp = self.timestamp.astimezone(timezone.utc)
        client_id = _validated_scope("client_id", self.client_id)
        environment = _validated_scope("environment", self.environment)
        runtime_mode = _required_text("runtime_mode", self.runtime_mode, 64)

        object.__setattr__(self, "event_id", self.event_id.strip())
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "client_id", client_id)
        object.__setattr__(self, "environment", environment)
        object.__setattr__(self, "runtime_mode", runtime_mode)

        for name in ("session_id", "request_id"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str) or not _IDENTIFIER.fullmatch(
                    value.strip()
                ):
                    raise ValueError(f"{name} is invalid")
                object.__setattr__(self, name, value.strip())

        for name in (
            "intent",
            "retrieval_strategy",
            "prompt_strategy",
            "llm_provider",
            "model_id",
            "response_mode",
            "error_category",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _required_text(name, value, 160),
                )

        for name in (
            "latency_ms",
            "retrieval_latency_ms",
            "generation_latency_ms",
        ):
            _validate_non_negative_number(name, getattr(self, name))
        for name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "retrieved_source_count",
            "citation_count",
        ):
            _validate_non_negative_integer(name, getattr(self, name))
        for name in ("approval_required", "success"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean or null")

        if self.estimated_cost_usd is not None:
            cost = _decimal("estimated_cost_usd", self.estimated_cost_usd)
            object.__setattr__(self, "estimated_cost_usd", cost)
        if (
            self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens is not None
            and self.input_tokens + self.output_tokens != self.total_tokens
        ):
            raise ValueError("total_tokens must equal input plus output")

        if self.safety_outcome is not None:
            try:
                outcome = SafetyOutcome(self.safety_outcome)
            except (TypeError, ValueError) as error:
                raise ValueError("safety_outcome is invalid") from error
            object.__setattr__(self, "safety_outcome", outcome)
        if self.user_rating is not None:
            try:
                rating = UserRating(self.user_rating)
            except (TypeError, ValueError) as error:
                raise ValueError("user_rating is invalid") from error
            object.__setattr__(self, "user_rating", rating)

        if self.feedback_text is not None:
            feedback = _required_text(
                "feedback_text",
                self.feedback_text,
                500,
            )
            _reject_sensitive_value("feedback_text", feedback)
            object.__setattr__(self, "feedback_text", feedback)

        metadata = _safe_metadata(self.evaluation_metadata)
        object.__setattr__(self, "evaluation_metadata", metadata)
        self._validate_event_specific_fields()

    def _validate_event_specific_fields(self) -> None:
        if self.event_type == MonitoringEventType.APPLICATION_REQUEST:
            _require(self.request_id, "Application requests need request_id")
            _require(self.session_id, "Application requests need session_id")
        elif self.event_type == MonitoringEventType.RETRIEVAL_COMPLETION:
            _require(
                self.retrieval_strategy,
                "Retrieval events need retrieval_strategy",
            )
            _require(self.request_id, "Retrieval events need request_id")
        elif self.event_type == MonitoringEventType.LLM_COMPLETION:
            _require(self.prompt_strategy, "LLM events need prompt_strategy")
            _require(self.llm_provider, "LLM events need llm_provider")
            _require(self.model_id, "LLM events need model_id")
            _require(self.request_id, "LLM events need request_id")
        elif self.event_type == MonitoringEventType.SAFETY_DECISION:
            _require(
                self.safety_outcome,
                "Safety events need safety_outcome",
            )
            _require(self.request_id, "Safety events need request_id")
        elif self.event_type == MonitoringEventType.APPROVAL_REQUIREMENT:
            if self.approval_required is not True:
                raise ValueError(
                    "Approval events need approval_required=true"
                )
            _require(self.request_id, "Approval events need request_id")
        elif self.event_type == MonitoringEventType.USER_FEEDBACK:
            _require(self.user_rating, "Feedback events need user_rating")
            _require(self.request_id, "Feedback events need request_id")
        elif self.event_type == MonitoringEventType.EVALUATION_RUN:
            if not self.evaluation_metadata:
                raise ValueError(
                    "Evaluation events need evaluation_metadata"
                )
        elif self.event_type == MonitoringEventType.APPLICATION_ERROR:
            if self.success is not False:
                raise ValueError("Application errors need success=false")
            _require(
                self.error_category,
                "Application errors need error_category",
            )
            _require(self.request_id, "Application errors need request_id")

    def to_dict(self) -> dict[str, Any]:
        """Serialize with stable field names and Decimal-safe values."""

        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": _timestamp_text(self.timestamp),
            "session_id": self.session_id,
            "request_id": self.request_id,
            "client_id": self.client_id,
            "environment": self.environment,
            "runtime_mode": self.runtime_mode,
            "intent": self.intent,
            "retrieval_strategy": self.retrieval_strategy,
            "prompt_strategy": self.prompt_strategy,
            "llm_provider": self.llm_provider,
            "model_id": self.model_id,
            "response_mode": self.response_mode,
            "latency_ms": self.latency_ms,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": (
                format(self.estimated_cost_usd, "f")
                if self.estimated_cost_usd is not None
                else None
            ),
            "retrieved_source_count": self.retrieved_source_count,
            "citation_count": self.citation_count,
            "approval_required": self.approval_required,
            "safety_outcome": (
                self.safety_outcome.value
                if self.safety_outcome is not None
                else None
            ),
            "success": self.success,
            "error_category": self.error_category,
            "user_rating": (
                self.user_rating.value
                if self.user_rating is not None
                else None
            ),
            "feedback_text": self.feedback_text,
            "evaluation_metadata": _json_safe(self.evaluation_metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MonitoringEvent:
        """Deserialize and validate one schema-versioned record."""

        if not isinstance(payload, Mapping):
            raise ValueError("Monitoring record must be an object")
        known = set(cls.__dataclass_fields__)
        unknown = set(payload) - known
        if unknown:
            raise ValueError(
                "Monitoring record contains unknown fields: "
                + ", ".join(sorted(unknown))
            )
        values = dict(payload)
        raw_timestamp = values.get("timestamp")
        if not isinstance(raw_timestamp, str):
            raise ValueError("timestamp must be an ISO string")
        try:
            values["timestamp"] = datetime.fromisoformat(
                raw_timestamp.replace("Z", "+00:00")
            )
        except ValueError as error:
            raise ValueError("timestamp must be a valid ISO string") from error
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError(
                "Monitoring record is missing required fields"
            ) from error


@dataclass(frozen=True)
class MalformedMonitoringRecord:
    """Safe parse failure without retaining the malformed raw record."""

    line_number: int
    error: str


@dataclass(frozen=True)
class MonitoringLoadResult:
    """Events plus non-sensitive details for skipped malformed records."""

    events: tuple[MonitoringEvent, ...]
    malformed_records: tuple[MalformedMonitoringRecord, ...]


class JsonLinesEventSink:
    """Append and read scoped monitoring events from a local JSONL file."""

    def __init__(
        self,
        path: Path | str = DEFAULT_MONITORING_PATH,
    ) -> None:
        self.path = Path(path)
        if not self.path.name or self.path.suffix.casefold() != ".jsonl":
            raise ValueError("Monitoring sink path must end in .jsonl")

    def append(self, event: MonitoringEvent) -> None:
        """Append one event without rewriting existing records."""

        self.append_many((event,))

    def append_many(self, events: Sequence[MonitoringEvent]) -> None:
        """Append validated events in caller-provided order."""

        records = tuple(events)
        if any(not isinstance(event, MonitoringEvent) for event in records):
            raise ValueError("events must contain MonitoringEvent values")
        if not records:
            return
        serialized = tuple(
            json.dumps(
                event.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            for event in records
        )
        if any(
            len(record) > MAXIMUM_RECORD_CHARACTERS
            for record in serialized
        ):
            raise ValueError("Monitoring record exceeds maximum size")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            for record in serialized:
                stream.write(record + "\n")

    def load(
        self,
        *,
        client_id: str | None = None,
        environment: str | None = None,
        skip_malformed: bool = True,
    ) -> MonitoringLoadResult:
        """Read events, optionally filtering one exact client scope."""

        if client_id is not None:
            client_id = _validated_scope("client_id", client_id)
        if environment is not None:
            environment = _validated_scope("environment", environment)
        if not self.path.exists():
            return MonitoringLoadResult((), ())
        events: list[MonitoringEvent] = []
        malformed: list[MalformedMonitoringRecord] = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.rstrip("\r\n")
                try:
                    if not line:
                        raise ValueError("record is empty")
                    if len(line) > MAXIMUM_RECORD_CHARACTERS:
                        raise ValueError("record exceeds maximum size")
                    payload = json.loads(line)
                    event = MonitoringEvent.from_dict(payload)
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    if not skip_malformed:
                        raise ValueError(
                            f"Malformed monitoring record at line "
                            f"{line_number}: {error}"
                        ) from error
                    malformed.append(
                        MalformedMonitoringRecord(
                            line_number=line_number,
                            error=str(error),
                        )
                    )
                    continue
                if client_id is not None and event.client_id != client_id:
                    continue
                if (
                    environment is not None
                    and event.environment != environment
                ):
                    continue
                events.append(event)
        return MonitoringLoadResult(
            events=tuple(events),
            malformed_records=tuple(malformed),
        )


def _validated_scope(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} is invalid")
    normalized = value.strip().lower()
    if not _SCOPE.fullmatch(normalized):
        raise ValueError(f"{name} is invalid")
    return normalized


def _required_text(name: str, value: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    return normalized


def _validate_non_negative_number(
    name: str,
    value: float | None,
) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative")


def _validate_non_negative_integer(
    name: str,
    value: int | None,
) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")


def _decimal(name: str, value: Any) -> Decimal:
    if isinstance(value, (float, bool)):
        raise ValueError(f"{name} must use an exact decimal value")
    try:
        result = Decimal(value)
    except (InvalidOperation, TypeError) as error:
        raise ValueError(f"{name} must be a valid decimal") from error
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _safe_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("evaluation_metadata must be a mapping")
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("evaluation_metadata keys must be strings")
        clean_key = key.strip()
        if clean_key.casefold() in _SENSITIVE_KEYS:
            raise ValueError(
                f"evaluation_metadata field is sensitive: {clean_key}"
            )
        normalized[clean_key] = _safe_metadata_value(
            item,
            path=f"evaluation_metadata.{clean_key}",
        )
    return normalized


def _safe_metadata_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return value
    if isinstance(value, Decimal):
        return format(_decimal(path, value), "f")
    if isinstance(value, str):
        if len(value) > 500:
            raise ValueError(f"{path} exceeds 500 characters")
        _reject_sensitive_value(path, value)
        return value
    if isinstance(value, Mapping):
        return _safe_metadata(value)
    if isinstance(value, (list, tuple)):
        return [
            _safe_metadata_value(item, path=f"{path}[]")
            for item in value
        ]
    raise ValueError(f"{path} contains an unsupported value")


def _reject_sensitive_value(name: str, value: str) -> None:
    if any(pattern.search(value) for pattern in _SENSITIVE_VALUE_PATTERNS):
        raise ValueError(f"{name} appears to contain sensitive data")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, Mapping):
        return {
            key: _json_safe(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _timestamp_text(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _require(value: Any, message: str) -> None:
    if value is None:
        raise ValueError(message)
