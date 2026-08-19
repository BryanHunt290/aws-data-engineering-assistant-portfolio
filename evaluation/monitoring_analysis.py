"""Offline monitoring and feedback aggregation without external services."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import math
import platform
import subprocess
from typing import Any, Callable, Mapping, Sequence

from knowledge.monitoring import (
    MonitoringEvent,
    MonitoringEventType,
    SafetyOutcome,
    UserRating,
)


MONITORING_ANALYSIS_VERSION = "offline-monitoring-analysis-v1"
DEFAULT_HIGH_COST_THRESHOLD_USD = Decimal("0.001")
DEFAULT_SLOW_REQUEST_THRESHOLD_MS = 1000.0


@dataclass(frozen=True)
class MonitoringAnalysisConfig:
    """Thresholds used for transparent synthetic alert indicators."""

    high_cost_threshold_usd: Decimal = DEFAULT_HIGH_COST_THRESHOLD_USD
    slow_request_threshold_ms: float = DEFAULT_SLOW_REQUEST_THRESHOLD_MS

    def __post_init__(self) -> None:
        if (
            not isinstance(self.high_cost_threshold_usd, Decimal)
            or not self.high_cost_threshold_usd.is_finite()
            or self.high_cost_threshold_usd < 0
        ):
            raise ValueError(
                "high_cost_threshold_usd must be a non-negative Decimal"
            )
        if (
            isinstance(self.slow_request_threshold_ms, bool)
            or not isinstance(self.slow_request_threshold_ms, (int, float))
            or not math.isfinite(self.slow_request_threshold_ms)
            or self.slow_request_threshold_ms <= 0
        ):
            raise ValueError(
                "slow_request_threshold_ms must be finite and positive"
            )


@dataclass(frozen=True)
class MonitoringAnalysis:
    """Reviewer-facing aggregate metrics with no raw feedback or prompts."""

    metadata: dict[str, Any]
    thresholds: dict[str, Any]
    overview: dict[str, Any]
    by_event_type: dict[str, int]
    by_retrieval_strategy: dict[str, dict[str, Any]]
    by_prompt_strategy: dict[str, dict[str, Any]]
    by_response_mode: dict[str, dict[str, Any]]
    by_intent: dict[str, dict[str, Any]]
    by_runtime_mode: dict[str, dict[str, Any]]
    by_safety_outcome: dict[str, dict[str, Any]]
    by_client: dict[str, dict[str, Any]]
    by_environment: dict[str, dict[str, Any]]
    by_scope: dict[str, dict[str, Any]]
    daily: tuple[dict[str, Any], ...]
    error_categories: dict[str, int]
    evaluation_runs: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe deterministic representation."""

        return {
            "schema_version": 1,
            "metadata": _json_safe(self.metadata),
            "thresholds": _json_safe(self.thresholds),
            "overview": _json_safe(self.overview),
            "by_event_type": self.by_event_type,
            "by_retrieval_strategy": _json_safe(
                self.by_retrieval_strategy
            ),
            "by_prompt_strategy": _json_safe(self.by_prompt_strategy),
            "by_response_mode": _json_safe(self.by_response_mode),
            "by_intent": _json_safe(self.by_intent),
            "by_runtime_mode": _json_safe(self.by_runtime_mode),
            "by_safety_outcome": _json_safe(self.by_safety_outcome),
            "by_client": _json_safe(self.by_client),
            "by_environment": _json_safe(self.by_environment),
            "by_scope": _json_safe(self.by_scope),
            "daily": _json_safe(self.daily),
            "error_categories": self.error_categories,
            "evaluation_runs": _json_safe(self.evaluation_runs),
        }


def analyze_monitoring_events(
    events: Sequence[MonitoringEvent],
    *,
    config: MonitoringAnalysisConfig | None = None,
    evaluated_at: str | None = None,
    git_commit: str | None = None,
    malformed_record_count: int = 0,
) -> MonitoringAnalysis:
    """Aggregate scoped events using deterministic arithmetic and joins."""

    settings = config or MonitoringAnalysisConfig()
    records = tuple(events)
    if not records:
        raise ValueError("At least one monitoring event is required")
    if any(not isinstance(event, MonitoringEvent) for event in records):
        raise ValueError("events must contain MonitoringEvent values")
    identifiers = [event.event_id for event in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Monitoring event IDs must be unique")
    if (
        isinstance(malformed_record_count, bool)
        or not isinstance(malformed_record_count, int)
        or malformed_record_count < 0
    ):
        raise ValueError(
            "malformed_record_count must be a non-negative integer"
        )
    timestamp = evaluated_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    _validate_timestamp(timestamp)

    requests = _events(records, MonitoringEventType.APPLICATION_REQUEST)
    retrievals = _events(
        records,
        MonitoringEventType.RETRIEVAL_COMPLETION,
    )
    completions = _events(records, MonitoringEventType.LLM_COMPLETION)
    feedback = _events(records, MonitoringEventType.USER_FEEDBACK)
    errors = _events(records, MonitoringEventType.APPLICATION_ERROR)
    safety = _events(records, MonitoringEventType.SAFETY_DECISION)
    approvals = _events(
        records,
        MonitoringEventType.APPROVAL_REQUIREMENT,
    )
    evaluations = _events(records, MonitoringEventType.EVALUATION_RUN)
    feedback_by_request = {
        _request_key(event): event
        for event in feedback
        if event.request_id is not None
    }
    errors_by_request = {
        _request_key(event)
        for event in errors
        if event.request_id is not None
    }
    request_keys = {_request_key(event) for event in requests}
    errored_request_count = len(request_keys & errors_by_request)
    completion_costs = [
        event.estimated_cost_usd
        for event in completions
        if event.estimated_cost_usd is not None
    ]
    request_latencies = [
        event.latency_ms
        for event in completions
        if event.latency_ms is not None
    ]
    retrieval_latencies = [
        event.retrieval_latency_ms
        for event in retrievals
        if event.retrieval_latency_ms is not None
    ]
    generation_latencies = [
        event.generation_latency_ms
        for event in completions
        if event.generation_latency_ms is not None
    ]
    grounded = [
        event
        for event in completions
        if (event.retrieved_source_count or 0) > 0
    ]
    successful_grounded = [
        event
        for event in grounded
        if event.success is True and (event.citation_count or 0) > 0
    ]
    complete_citations = [
        event
        for event in grounded
        if event.citation_count == event.retrieved_source_count
    ]
    positive_feedback = [
        event for event in feedback if event.user_rating == UserRating.UP
    ]
    negative_feedback = [
        event for event in feedback if event.user_rating == UserRating.DOWN
    ]
    no_results = [
        event
        for event in retrievals
        if event.retrieved_source_count == 0
    ]
    high_cost = [
        event
        for event in completions
        if event.estimated_cost_usd is not None
        and event.estimated_cost_usd
        >= settings.high_cost_threshold_usd
    ]
    slow = [
        event
        for event in completions
        if event.latency_ms is not None
        and event.latency_ms >= settings.slow_request_threshold_ms
    ]
    blocked = [
        event
        for event in safety
        if event.safety_outcome
        in {
            SafetyOutcome.BLOCKED,
            SafetyOutcome.SAFETY_REVIEW_REQUIRED,
        }
    ]
    request_count = len(requests)
    overview = {
        "event_count": len(records),
        "request_count": request_count,
        "session_count": len(
            {event.session_id for event in records if event.session_id}
        ),
        "scope_count": len(
            {(event.client_id, event.environment) for event in records}
        ),
        "llm_completion_count": len(completions),
        "request_success_rate": _ratio(
            request_count - errored_request_count,
            request_count,
        ),
        "application_error_rate": _ratio(
            errored_request_count,
            request_count,
        ),
        "grounded_response_rate": _ratio(
            len(successful_grounded),
            len(grounded),
        ),
        "citation_completeness_rate": _ratio(
            len(complete_citations),
            len(grounded),
        ),
        "no_result_rate": _ratio(len(no_results), len(retrievals)),
        "feedback_coverage_rate": _ratio(
            len(feedback_by_request),
            request_count,
        ),
        "feedback_rate": _ratio(len(feedback_by_request), request_count),
        "positive_feedback_rate": _ratio(
            len(positive_feedback),
            len(feedback),
        ),
        "negative_feedback_rate": _ratio(
            len(negative_feedback),
            len(feedback),
        ),
        "average_rating": _ratio(
            len(positive_feedback),
            len(feedback),
        ),
        "approval_requirement_rate": _ratio(
            len(approvals),
            request_count,
        ),
        "safety_block_rate": _ratio(len(blocked), request_count),
        "safety_event_count": len(safety),
        "approval_required_count": len(approvals),
        "high_cost_request_rate": _ratio(
            len(high_cost),
            len(completions),
        ),
        "slow_request_rate": _ratio(len(slow), len(completions)),
        "total_input_tokens": sum(
            event.input_tokens or 0 for event in completions
        ),
        "total_output_tokens": sum(
            event.output_tokens or 0 for event in completions
        ),
        "total_tokens": sum(
            event.total_tokens or 0 for event in completions
        ),
        "average_total_tokens": _mean(
            [
                float(event.total_tokens)
                for event in completions
                if event.total_tokens is not None
            ]
        ),
        "average_input_tokens": _mean(
            [
                float(event.input_tokens)
                for event in completions
                if event.input_tokens is not None
            ]
        ),
        "average_output_tokens": _mean(
            [
                float(event.output_tokens)
                for event in completions
                if event.output_tokens is not None
            ]
        ),
        "total_estimated_cost_usd": sum(
            completion_costs,
            Decimal("0"),
        ),
        "average_estimated_cost_usd": (
            sum(completion_costs, Decimal("0"))
            / Decimal(len(completion_costs))
            if completion_costs
            else Decimal("0")
        ),
        "average_estimated_cost_per_request_usd": (
            sum(completion_costs, Decimal("0")) / Decimal(request_count)
            if request_count
            else Decimal("0")
        ),
        "average_latency_ms": _mean(request_latencies),
        "p50_latency_ms": _percentile_or_zero(request_latencies, 0.50),
        "p95_latency_ms": _percentile_or_zero(request_latencies, 0.95),
        "average_retrieval_latency_ms": _mean(retrieval_latencies),
        "average_generation_latency_ms": _mean(generation_latencies),
        "request_latency_ms": _latency_summary(request_latencies),
        "retrieval_latency_ms": _latency_summary(retrieval_latencies),
        "generation_latency_ms": _latency_summary(
            generation_latencies
        ),
    }

    dataset_versions = sorted(
        {
            str(event.evaluation_metadata["dataset_version"])
            for event in records
            if "dataset_version" in event.evaluation_metadata
        }
    )
    is_synthetic = all(
        event.evaluation_metadata.get("synthetic") is True
        for event in records
    )
    return MonitoringAnalysis(
        metadata={
            "analysis_version": MONITORING_ANALYSIS_VERSION,
            "evaluation_date": timestamp,
            "git_commit": git_commit or _git_commit(),
            "python_version": platform.python_version(),
            "dataset_versions": dataset_versions,
            "event_schema_version": records[0].schema_version,
            "data_classification": (
                "synthetic" if is_synthetic else "unclassified-local"
            ),
            "malformed_record_count": malformed_record_count,
            "network_calls": False,
            "provider_charges_incurred": False,
            "estimated_cost_is_simulated": is_synthetic,
            "raw_prompts_included": False,
            "raw_document_text_included": False,
            "raw_feedback_included_in_report": False,
        },
        thresholds={
            "high_cost_threshold_usd": settings.high_cost_threshold_usd,
            "slow_request_threshold_ms": (
                settings.slow_request_threshold_ms
            ),
        },
        overview=_rounded(overview),
        by_event_type=dict(
            sorted(Counter(event.event_type.value for event in records).items())
        ),
        by_retrieval_strategy=_group_retrieval(retrievals, completions),
        by_prompt_strategy=_group_prompt(
            completions,
            feedback_by_request,
        ),
        by_response_mode=_group_completion(
            completions,
            lambda event: event.response_mode or "unspecified",
        ),
        by_intent=_group_intent(
            requests,
            feedback_by_request,
            errors_by_request,
        ),
        by_runtime_mode=_group_dimension(
            records,
            lambda event: event.runtime_mode,
        ),
        by_safety_outcome=_group_dimension(
            tuple(
                event for event in records if event.safety_outcome is not None
            ),
            lambda event: (
                event.safety_outcome.value
                if event.safety_outcome is not None
                else "unspecified"
            ),
        ),
        by_client=_group_dimension(
            records,
            lambda event: event.client_id,
        ),
        by_environment=_group_dimension(
            records,
            lambda event: event.environment,
        ),
        by_scope=_group_scope(records),
        daily=_daily(records),
        error_categories=dict(
            sorted(
                Counter(
                    event.error_category or "unspecified"
                    for event in errors
                ).items()
            )
        ),
        evaluation_runs=tuple(
            {
                "event_id": event.event_id,
                "timestamp": event.to_dict()["timestamp"],
                "client_id": event.client_id,
                "environment": event.environment,
                "evaluation_name": event.evaluation_metadata.get(
                    "evaluation_name",
                    "unspecified",
                ),
                "success": event.success,
            }
            for event in evaluations
        ),
    )


def _group_retrieval(
    events: Sequence[MonitoringEvent],
    completions: Sequence[MonitoringEvent],
) -> dict[str, dict[str, Any]]:
    strategies = sorted(
        {event.retrieval_strategy or "unspecified" for event in events}
    )
    result: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        group = [
            event
            for event in events
            if (event.retrieval_strategy or "unspecified") == strategy
        ]
        latencies = [
            event.retrieval_latency_ms
            for event in group
            if event.retrieval_latency_ms is not None
        ]
        completion_group = [
            event
            for event in completions
            if (event.retrieval_strategy or "unspecified") == strategy
        ]
        costs = [
            event.estimated_cost_usd
            for event in completion_group
            if event.estimated_cost_usd is not None
        ]
        result[strategy] = _rounded(
            {
                "completion_count": len(group),
                "success_rate": _ratio(
                    sum(event.success is True for event in group),
                    len(group),
                ),
                "no_result_rate": _ratio(
                    sum(event.retrieved_source_count == 0 for event in group),
                    len(group),
                ),
                "average_source_count": _mean(
                    [
                        float(event.retrieved_source_count or 0)
                        for event in group
                    ]
                ),
                "latency_ms": _latency_summary(latencies),
                "average_estimated_cost_usd": (
                    sum(costs, Decimal("0")) / Decimal(len(costs))
                    if costs
                    else Decimal("0")
                ),
                "average_total_tokens": _mean(
                    [
                        float(event.total_tokens)
                        for event in completion_group
                        if event.total_tokens is not None
                    ]
                ),
            }
        )
    return result


def _group_prompt(
    events: Sequence[MonitoringEvent],
    feedback_by_request: Mapping[
        tuple[str, str, str] | None,
        MonitoringEvent,
    ],
) -> dict[str, dict[str, Any]]:
    strategies = sorted(
        {event.prompt_strategy or "unspecified" for event in events}
    )
    result: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        group = [
            event
            for event in events
            if (event.prompt_strategy or "unspecified") == strategy
        ]
        grounded = [
            event
            for event in group
            if (event.retrieved_source_count or 0) > 0
        ]
        feedback = [
            feedback_by_request[_request_key(event)]
            for event in group
            if _request_key(event) in feedback_by_request
        ]
        costs = [
            event.estimated_cost_usd
            for event in group
            if event.estimated_cost_usd is not None
        ]
        result[strategy] = _rounded(
            {
                "completion_count": len(group),
                "grounded_response_rate": _ratio(
                    sum(
                        event.success is True
                        and (event.citation_count or 0) > 0
                        for event in grounded
                    ),
                    len(grounded),
                ),
                "citation_completeness_rate": _ratio(
                    sum(
                        event.citation_count
                        == event.retrieved_source_count
                        for event in grounded
                    ),
                    len(grounded),
                ),
                "feedback_count": len(feedback),
                "positive_feedback_rate": _ratio(
                    sum(
                        event.user_rating == UserRating.UP
                        for event in feedback
                    ),
                    len(feedback),
                ),
                "average_input_tokens": _mean(
                    [
                        float(event.input_tokens)
                        for event in group
                        if event.input_tokens is not None
                    ]
                ),
                "average_output_tokens": _mean(
                    [
                        float(event.output_tokens)
                        for event in group
                        if event.output_tokens is not None
                    ]
                ),
                "average_total_tokens": _mean(
                    [
                        float(event.total_tokens)
                        for event in group
                        if event.total_tokens is not None
                    ]
                ),
                "total_estimated_cost_usd": sum(
                    costs,
                    Decimal("0"),
                ),
                "average_estimated_cost_usd": (
                    sum(costs, Decimal("0")) / Decimal(len(costs))
                    if costs
                    else Decimal("0")
                ),
                "generation_latency_ms": _latency_summary(
                    [
                        event.generation_latency_ms
                        for event in group
                        if event.generation_latency_ms is not None
                    ]
                ),
                "latency_ms": _latency_summary(
                    [
                        event.latency_ms
                        for event in group
                        if event.latency_ms is not None
                    ]
                ),
            }
        )
    return result


def _group_completion(
    events: Sequence[MonitoringEvent],
    key: Callable[[MonitoringEvent], str],
) -> dict[str, dict[str, Any]]:
    names = sorted({key(event) for event in events})
    return {
        name: _rounded(
            {
                "completion_count": len(group),
                "average_total_tokens": _mean(
                    [
                        float(event.total_tokens)
                        for event in group
                        if event.total_tokens is not None
                    ]
                ),
                "average_estimated_cost_usd": (
                    sum(
                        (
                            event.estimated_cost_usd
                            for event in group
                            if event.estimated_cost_usd is not None
                        ),
                        Decimal("0"),
                    )
                    / Decimal(
                        max(
                            1,
                            sum(
                                event.estimated_cost_usd is not None
                                for event in group
                            ),
                        )
                    )
                ),
                "latency_ms": _latency_summary(
                    [
                        event.latency_ms
                        for event in group
                        if event.latency_ms is not None
                    ]
                ),
            }
        )
        for name in names
        for group in ([event for event in events if key(event) == name],)
    }


def _group_intent(
    requests: Sequence[MonitoringEvent],
    feedback_by_request: Mapping[
        tuple[str, str, str] | None,
        MonitoringEvent,
    ],
    errors_by_request: set[tuple[str, str, str] | None],
) -> dict[str, dict[str, Any]]:
    intents = sorted({event.intent or "unspecified" for event in requests})
    result: dict[str, dict[str, Any]] = {}
    for intent in intents:
        group = [
            event
            for event in requests
            if (event.intent or "unspecified") == intent
        ]
        feedback = [
            feedback_by_request[_request_key(event)]
            for event in group
            if _request_key(event) in feedback_by_request
        ]
        result[intent] = _rounded(
            {
                "request_count": len(group),
                "application_error_rate": _ratio(
                    sum(
                        _request_key(event) in errors_by_request
                        for event in group
                    ),
                    len(group),
                ),
                "feedback_count": len(feedback),
                "positive_feedback_rate": _ratio(
                    sum(
                        event.user_rating == UserRating.UP
                        for event in feedback
                    ),
                    len(feedback),
                ),
            }
        )
    return result


def _group_scope(
    events: Sequence[MonitoringEvent],
) -> dict[str, dict[str, Any]]:
    scopes = sorted(
        {(event.client_id, event.environment) for event in events}
    )
    return {
        f"{client_id}/{environment}": {
            "event_count": len(group),
            "request_count": sum(
                event.event_type == MonitoringEventType.APPLICATION_REQUEST
                for event in group
            ),
            "error_count": sum(
                event.event_type == MonitoringEventType.APPLICATION_ERROR
                for event in group
            ),
            "feedback_count": sum(
                event.event_type == MonitoringEventType.USER_FEEDBACK
                for event in group
            ),
        }
        for client_id, environment in scopes
        for group in (
            [
                event
                for event in events
                if event.client_id == client_id
                and event.environment == environment
            ],
        )
    }


def _group_dimension(
    events: Sequence[MonitoringEvent],
    key: Callable[[MonitoringEvent], str],
) -> dict[str, dict[str, Any]]:
    """Aggregate common event measures for one requested dimension."""

    names = sorted({key(event) for event in events})
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        group = [event for event in events if key(event) == name]
        outcomes = [event.success for event in group if event.success is not None]
        result[name] = _rounded(
            {
                "event_count": len(group),
                "request_count": sum(
                    event.event_type
                    == MonitoringEventType.APPLICATION_REQUEST
                    for event in group
                ),
                "error_count": sum(
                    event.event_type == MonitoringEventType.APPLICATION_ERROR
                    for event in group
                ),
                "feedback_count": sum(
                    event.event_type == MonitoringEventType.USER_FEEDBACK
                    for event in group
                ),
                "event_success_rate": _ratio(
                    sum(outcome is True for outcome in outcomes),
                    len(outcomes),
                ),
            }
        )
    return result


def _daily(
    events: Sequence[MonitoringEvent],
) -> tuple[dict[str, Any], ...]:
    groups: dict[str, list[MonitoringEvent]] = defaultdict(list)
    for event in events:
        groups[event.timestamp.date().isoformat()].append(event)
    rows: list[dict[str, Any]] = []
    for day in sorted(groups):
        group = groups[day]
        costs = [
            event.estimated_cost_usd
            for event in group
            if event.estimated_cost_usd is not None
        ]
        rows.append(
            {
                "date": day,
                "event_count": len(group),
                "request_count": sum(
                    event.event_type
                    == MonitoringEventType.APPLICATION_REQUEST
                    for event in group
                ),
                "llm_completion_count": sum(
                    event.event_type == MonitoringEventType.LLM_COMPLETION
                    for event in group
                ),
                "error_count": sum(
                    event.event_type
                    == MonitoringEventType.APPLICATION_ERROR
                    for event in group
                ),
                "feedback_count": sum(
                    event.event_type == MonitoringEventType.USER_FEEDBACK
                    for event in group
                ),
                "positive_feedback_count": sum(
                    event.event_type == MonitoringEventType.USER_FEEDBACK
                    and event.user_rating == UserRating.UP
                    for event in group
                ),
                "estimated_cost_usd": sum(costs, Decimal("0")),
            }
        )
    return tuple(rows)


def _events(
    events: Sequence[MonitoringEvent],
    event_type: MonitoringEventType,
) -> tuple[MonitoringEvent, ...]:
    return tuple(event for event in events if event.event_type == event_type)


def _request_key(
    event: MonitoringEvent,
) -> tuple[str, str, str] | None:
    if event.request_id is None:
        return None
    return (event.client_id, event.environment, event.request_id)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "average": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "maximum": 0.0,
        }
    return {
        "average": _mean(ordered),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "maximum": ordered[-1],
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    return values[max(0, math.ceil(percentile * len(values)) - 1)]


def _percentile_or_zero(
    values: Sequence[float],
    percentile: float,
) -> float:
    ordered = sorted(float(value) for value in values)
    return _percentile(ordered, percentile) if ordered else 0.0


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    return value


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


def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError("evaluated_at must be a valid ISO timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("evaluated_at must include a timezone")


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"
