"""Deterministic synthetic monitoring data for offline demonstrations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import random

from knowledge.monitoring import (
    JsonLinesEventSink,
    MonitoringEvent,
    MonitoringEventType,
    SafetyOutcome,
    UserRating,
)


SYNTHETIC_DATASET_VERSION = "synthetic-monitoring-v1"
DEFAULT_SYNTHETIC_EVENT_PATH = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "fixtures"
    / "monitoring_events.jsonl"
)
DEFAULT_REQUEST_COUNT = 84
MINIMUM_SYNTHETIC_EVENTS = 250
SYNTHETIC_RANDOM_SEED = 20260727
_START = datetime(2026, 7, 1, 8, 0, tzinfo=timezone.utc)
_RETRIEVAL_STRATEGIES = ("semantic", "keyword", "hybrid")
_PROMPT_STRATEGIES = (
    "baseline-concise",
    "grounded-evidence-first",
    "structured-troubleshooting",
)
_RESPONSE_MODES = ("concise", "detailed")
_SCOPES = (
    ("demo-client-a", "dev"),
    ("demo-client-b", "test"),
)
_INTENTS = (
    "knowledge_question",
    "pipeline_troubleshooting",
    "architecture_design",
    "sql_generation",
    "pyspark_generation",
    "iam_review",
    "monitoring_request",
    "cost_question",
)


def generate_synthetic_monitoring_events(
    *,
    request_count: int = DEFAULT_REQUEST_COUNT,
    random_seed: int = SYNTHETIC_RANDOM_SEED,
) -> tuple[MonitoringEvent, ...]:
    """Create a stable multi-scope dataset from an explicit fixed seed."""

    if (
        isinstance(request_count, bool)
        or not isinstance(request_count, int)
        or request_count < 80
    ):
        raise ValueError("request_count must be an integer of at least 80")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer")

    randomizer = random.Random(random_seed)
    events: list[MonitoringEvent] = []
    event_number = 0

    def add(
        event_type: MonitoringEventType,
        timestamp: datetime,
        client_id: str,
        environment: str,
        **values,
    ) -> None:
        nonlocal event_number
        event_number += 1
        events.append(
            MonitoringEvent(
                event_id=f"synthetic-event-{event_number:04d}",
                event_type=event_type,
                timestamp=timestamp,
                client_id=client_id,
                environment=environment,
                runtime_mode="offline-synthetic",
                **values,
            )
        )

    for index in range(1, request_count + 1):
        base_time = _START + timedelta(hours=(index - 1) * 4)
        scope_index = (index - 1) % len(_SCOPES)
        client_id, environment = _SCOPES[scope_index]
        request_id = f"synthetic-request-{index:04d}"
        session_id = (
            f"synthetic-session-{scope_index + 1}-"
            f"{((index - 1) // 8) + 1:03d}"
        )
        retrieval_strategy = _RETRIEVAL_STRATEGIES[
            (index - 1) % len(_RETRIEVAL_STRATEGIES)
        ]
        prompt_strategy = _PROMPT_STRATEGIES[
            (
                (index - 1)
                + ((index - 1) // len(_PROMPT_STRATEGIES))
            )
            % len(_PROMPT_STRATEGIES)
        ]
        response_mode = _RESPONSE_MODES[
            ((index - 1) // len(_RESPONSE_MODES))
            % len(_RESPONSE_MODES)
        ]
        is_safety = index % 9 == 0
        is_approval = index % 8 == 0 and not is_safety
        is_error = index % 17 == 0 and not is_safety and not is_approval
        no_result = index % 10 == 0
        incomplete_citations = (
            (
                prompt_strategy == "baseline-concise"
                and index % 7 == 0
            )
            or (
                prompt_strategy == "structured-troubleshooting"
                and index % 13 == 0
            )
            or (
                prompt_strategy == "grounded-evidence-first"
                and index % 29 == 0
            )
        )
        high_cost = index % 14 == 0
        slow_request = index % 13 == 0
        if is_safety:
            intent = "destructive_action_request"
        elif is_approval:
            intent = "deployment_request"
        else:
            intent = _INTENTS[(index - 1) % len(_INTENTS)]
        scenario_flags = [
            name
            for name, enabled in (
                ("safety_sensitive", is_safety),
                ("approval_required", is_approval),
                ("application_error", is_error),
                ("no_result", no_result),
                ("incomplete_citations", incomplete_citations),
                ("higher_cost", high_cost),
                ("slow_request", slow_request),
            )
            if enabled
        ] or ["successful_grounded_response"]
        shared = {
            "session_id": session_id,
            "request_id": request_id,
            "intent": intent,
            "retrieval_strategy": retrieval_strategy,
            "prompt_strategy": prompt_strategy,
            "response_mode": response_mode,
            "approval_required": is_approval,
            "evaluation_metadata": {
                "dataset_version": SYNTHETIC_DATASET_VERSION,
                "random_seed": random_seed,
                "scenario_flags": scenario_flags,
                "synthetic": True,
            },
        }
        add(
            MonitoringEventType.APPLICATION_REQUEST,
            base_time,
            client_id,
            environment,
            success=True,
            **shared,
        )

        if is_safety:
            add(
                MonitoringEventType.SAFETY_DECISION,
                base_time + timedelta(seconds=1),
                client_id,
                environment,
                safety_outcome=SafetyOutcome.BLOCKED,
                success=True,
                **shared,
            )
        elif is_approval:
            add(
                MonitoringEventType.APPROVAL_REQUIREMENT,
                base_time + timedelta(seconds=1),
                client_id,
                environment,
                safety_outcome=SafetyOutcome.APPROVAL_REQUIRED,
                success=True,
                **shared,
            )
        elif is_error:
            add(
                MonitoringEventType.APPLICATION_ERROR,
                base_time + timedelta(seconds=2),
                client_id,
                environment,
                success=False,
                error_category=(
                    "provider_timeout"
                    if index % 2
                    else "malformed_provider_response"
                ),
                latency_ms=1800.0 if slow_request else 240.0,
                **shared,
            )
        else:
            source_count = 0 if no_result else 1 + (index % 5)
            citation_count = source_count
            if incomplete_citations and source_count > 0:
                citation_count -= 1
            retrieval_latency = (
                {"semantic": 48.0, "keyword": 27.0, "hybrid": 72.0}[
                    retrieval_strategy
                ]
                + (index % 7) * 2.5
                + randomizer.randrange(0, 6) * 0.5
                + (650.0 if slow_request else 0.0)
            )
            add(
                MonitoringEventType.RETRIEVAL_COMPLETION,
                base_time + timedelta(seconds=1),
                client_id,
                environment,
                retrieval_latency_ms=retrieval_latency,
                retrieved_source_count=source_count,
                success=True,
                **shared,
            )

            input_tokens = 220 + (index % 7) * 35
            output_tokens = 65 if response_mode == "concise" else 180
            input_tokens += {
                "baseline-concise": 0,
                "grounded-evidence-first": 80,
                "structured-troubleshooting": 110,
            }[prompt_strategy]
            output_tokens += {
                "baseline-concise": 0,
                "grounded-evidence-first": 30,
                "structured-troubleshooting": 60,
            }[prompt_strategy]
            if high_cost:
                input_tokens += 1800
                output_tokens += 800
            total_tokens = input_tokens + output_tokens
            cost = (
                Decimal(input_tokens) * Decimal("0.25")
                + Decimal(output_tokens) * Decimal("1.25")
            ) / Decimal("1000000")
            generation_latency = (
                (95.0 if response_mode == "concise" else 185.0)
                + (index % 9) * 7.0
                + (1250.0 if slow_request else 0.0)
            )
            add(
                MonitoringEventType.LLM_COMPLETION,
                base_time + timedelta(seconds=2),
                client_id,
                environment,
                llm_provider="deterministic-monitoring-fake",
                model_id="monitoring-synthetic-model-v1",
                generation_latency_ms=generation_latency,
                latency_ms=retrieval_latency + generation_latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=cost,
                retrieved_source_count=source_count,
                citation_count=citation_count,
                safety_outcome=SafetyOutcome.ALLOWED,
                success=True,
                **shared,
            )

        if index % 4 in {0, 1}:
            negative = is_error or index % 10 in {0, 1}
            add(
                MonitoringEventType.USER_FEEDBACK,
                base_time + timedelta(minutes=3),
                client_id,
                environment,
                user_rating=(
                    UserRating.DOWN if negative else UserRating.UP
                ),
                feedback_text=(
                    "Synthetic feedback: response needs clearer evidence."
                    if negative
                    else "Synthetic feedback: helpful scoped response."
                ),
                success=True,
                **shared,
            )

    final_time = _START + timedelta(hours=request_count * 4)
    for offset, evaluation_name in enumerate(
        ("retrieval-comparison", "prompt-comparison", "monitoring-analysis")
    ):
        client_id, environment = _SCOPES[offset % len(_SCOPES)]
        add(
            MonitoringEventType.EVALUATION_RUN,
            final_time + timedelta(minutes=offset),
            client_id,
            environment,
            success=True,
            evaluation_metadata={
                "dataset_version": SYNTHETIC_DATASET_VERSION,
                "evaluation_name": evaluation_name,
                "offline": True,
                "random_seed": random_seed,
                "synthetic": True,
            },
        )

    if len(events) < MINIMUM_SYNTHETIC_EVENTS:
        raise ValueError(
            "Synthetic generator did not produce the minimum event count"
        )
    return tuple(events)


def write_synthetic_monitoring_fixture(
    path: Path | str = DEFAULT_SYNTHETIC_EVENT_PATH,
    *,
    request_count: int = DEFAULT_REQUEST_COUNT,
    random_seed: int = SYNTHETIC_RANDOM_SEED,
    overwrite: bool = False,
) -> Path:
    """Write a reviewed fixture; overwrite requires an explicit opt-in."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            "Synthetic monitoring fixture already exists; use overwrite=True"
        )
    if destination.exists():
        destination.unlink()
    sink = JsonLinesEventSink(destination)
    sink.append_many(
        generate_synthetic_monitoring_events(
            request_count=request_count,
            random_seed=random_seed,
        )
    )
    return destination
