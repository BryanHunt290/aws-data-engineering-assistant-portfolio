from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import socket
import struct

import pytest
from streamlit.testing.v1 import AppTest

from evaluation.generate_monitoring_fixture import main as generate_main
from evaluation.monitoring_analysis import (
    MonitoringAnalysisConfig,
    analyze_monitoring_events,
)
from evaluation.monitoring_dataset import (
    DEFAULT_REQUEST_COUNT,
    MINIMUM_SYNTHETIC_EVENTS,
    SYNTHETIC_RANDOM_SEED,
    SYNTHETIC_DATASET_VERSION,
    generate_synthetic_monitoring_events,
    write_synthetic_monitoring_fixture,
)
from evaluation.monitoring_reporting import (
    CHART_DIRECTORY,
    CHART_FILENAMES,
    DAY_CSV_FILENAME,
    INTENT_CSV_FILENAME,
    JSON_FILENAME,
    MARKDOWN_FILENAME,
    STRATEGY_CSV_FILENAME,
    render_day_csv,
    render_intent_csv,
    render_monitoring_markdown,
    render_strategy_csv,
    write_monitoring_reports,
)
from evaluation.run_monitoring_report import main as analysis_main
from knowledge.monitoring import (
    CURRENT_MONITORING_SCHEMA_VERSION,
    JsonLinesEventSink,
    MonitoringEvent,
    MonitoringEventType,
    SafetyOutcome,
    UserRating,
)
from ui.monitoring_dashboard import load_monitoring_dashboard_data


FIXED_TIME = datetime(2026, 7, 1, tzinfo=timezone.utc)
FIXED_TIME_TEXT = "2026-07-27T00:00:00Z"
FIXED_COMMIT = "b" * 40


def _request_event(**overrides):
    values = {
        "event_id": "event-001",
        "event_type": MonitoringEventType.APPLICATION_REQUEST,
        "timestamp": FIXED_TIME,
        "client_id": "client-a",
        "environment": "dev",
        "runtime_mode": "offline",
        "session_id": "session-001",
        "request_id": "request-001",
        "intent": "knowledge_question",
        "success": True,
        "evaluation_metadata": {"synthetic": True},
    }
    values.update(overrides)
    return MonitoringEvent(**values)


def _analysis():
    return analyze_monitoring_events(
        generate_synthetic_monitoring_events(),
        evaluated_at=FIXED_TIME_TEXT,
        git_commit=FIXED_COMMIT,
    )


def test_monitoring_event_round_trip_normalizes_utc_and_decimal():
    event = _request_event(
        timestamp=datetime(
            2026,
            6,
            30,
            18,
            tzinfo=timezone(timedelta(hours=-6)),
        ),
        estimated_cost_usd=Decimal("0.0001200"),
        evaluation_metadata={
            "synthetic": True,
            "exact_decimal": Decimal("1.20"),
        },
    )

    payload = event.to_dict()
    restored = MonitoringEvent.from_dict(payload)

    assert payload["schema_version"] == CURRENT_MONITORING_SCHEMA_VERSION
    assert payload["timestamp"] == "2026-07-01T00:00:00Z"
    assert payload["estimated_cost_usd"] == "0.0001200"
    assert payload["evaluation_metadata"]["exact_decimal"] == "1.20"
    assert restored == event


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("timestamp", datetime(2026, 7, 1), "timezone"),
        ("event_id", "bad id", "event_id"),
        ("client_id", "Bad Client", "client_id"),
        ("latency_ms", -1.0, "latency_ms"),
        ("input_tokens", True, "input_tokens"),
        ("estimated_cost_usd", 0.1, "exact decimal"),
        ("schema_version", 99, "schema_version"),
    ],
)
def test_monitoring_event_rejects_invalid_core_fields(
    field,
    value,
    message,
):
    values = _request_event().to_dict()
    values["timestamp"] = FIXED_TIME
    values[field] = value

    with pytest.raises(ValueError, match=message):
        MonitoringEvent(**values)


def test_monitoring_event_rejects_inconsistent_token_total():
    with pytest.raises(ValueError, match="total_tokens"):
        _request_event(
            input_tokens=10,
            output_tokens=4,
            total_tokens=15,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "event_type": MonitoringEventType.RETRIEVAL_COMPLETION,
            "retrieval_strategy": None,
        },
        {
            "event_type": MonitoringEventType.LLM_COMPLETION,
            "prompt_strategy": None,
            "llm_provider": "fake",
            "model_id": "model",
        },
        {
            "event_type": MonitoringEventType.SAFETY_DECISION,
            "safety_outcome": None,
        },
        {
            "event_type": MonitoringEventType.APPROVAL_REQUIREMENT,
            "approval_required": False,
        },
        {
            "event_type": MonitoringEventType.USER_FEEDBACK,
            "user_rating": None,
        },
        {
            "event_type": MonitoringEventType.EVALUATION_RUN,
            "evaluation_metadata": {},
        },
        {
            "event_type": MonitoringEventType.APPLICATION_ERROR,
            "success": True,
            "error_category": None,
        },
    ],
)
def test_monitoring_event_enforces_event_specific_contracts(overrides):
    with pytest.raises(ValueError):
        _request_event(**overrides)


@pytest.mark.parametrize(
    "metadata",
    [
        {"prompt": "raw prompt"},
        {"nested": {"aws_secret_access_key": "not-allowed"}},
        {"value": "password=private-value"},
        {"value": "AKIA" + ("A" * 16)},
    ],
)
def test_monitoring_event_rejects_sensitive_metadata(metadata):
    with pytest.raises(ValueError, match="sensitive"):
        _request_event(evaluation_metadata=metadata)


def test_monitoring_event_rejects_sensitive_feedback_text():
    with pytest.raises(ValueError, match="sensitive"):
        _request_event(
            event_type=MonitoringEventType.USER_FEEDBACK,
            user_rating=UserRating.DOWN,
            feedback_text="authorization=Bearer-private",
        )


def test_jsonl_sink_appends_without_rewriting_and_serializes_stably(
    tmp_path,
):
    path = tmp_path / "events.jsonl"
    sink = JsonLinesEventSink(path)
    first = _request_event()
    second = replace(
        first,
        event_id="event-002",
        request_id="request-002",
        estimated_cost_usd=Decimal("0.0002"),
    )

    sink.append(first)
    original_line = path.read_text(encoding="utf-8")
    sink.append(second)
    lines = path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert lines[0] + "\n" == original_line
    assert lines[0] == json.dumps(
        first.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    assert json.loads(lines[1])["estimated_cost_usd"] == "0.0002"


def test_jsonl_sink_returns_empty_for_missing_file(tmp_path):
    result = JsonLinesEventSink(tmp_path / "missing.jsonl").load()

    assert result.events == ()
    assert result.malformed_records == ()


def test_jsonl_sink_skips_or_rejects_malformed_records(tmp_path):
    path = tmp_path / "events.jsonl"
    JsonLinesEventSink(path).append(_request_event())
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"schema_version": 1, "secret": "not retained"}\n')

    skipped = JsonLinesEventSink(path).load(skip_malformed=True)

    assert len(skipped.events) == 1
    assert len(skipped.malformed_records) == 1
    assert skipped.malformed_records[0].line_number == 2
    assert "not retained" not in skipped.malformed_records[0].error
    with pytest.raises(ValueError, match="line 2"):
        JsonLinesEventSink(path).load(skip_malformed=False)


def test_jsonl_sink_filters_exact_client_and_environment_scope(tmp_path):
    first = _request_event()
    second = replace(
        first,
        event_id="event-002",
        client_id="client-b",
        environment="test",
        request_id="request-002",
    )
    sink = JsonLinesEventSink(tmp_path / "events.jsonl")
    sink.append_many((first, second))

    result = sink.load(client_id="CLIENT-A", environment="dev")

    assert result.events == (first,)


def test_jsonl_sink_rejects_oversized_record_before_file_creation(
    tmp_path,
):
    path = tmp_path / "events.jsonl"
    event = _request_event(
        evaluation_metadata={
            "synthetic": True,
            "large_values": ["x" * 500] * 250,
        }
    )

    with pytest.raises(ValueError, match="maximum size"):
        JsonLinesEventSink(path).append(event)

    assert not path.exists()


def test_synthetic_dataset_is_deterministic_complete_and_multi_scope():
    first = generate_synthetic_monitoring_events()
    second = generate_synthetic_monitoring_events()
    event_types = {event.event_type for event in first}
    flags = {
        flag
        for event in first
        for flag in event.evaluation_metadata.get("scenario_flags", [])
    }

    assert first == second
    assert first != generate_synthetic_monitoring_events(
        random_seed=SYNTHETIC_RANDOM_SEED + 1
    )
    assert len(first) == 275
    assert len(first) >= MINIMUM_SYNTHETIC_EVENTS
    assert len(
        [
            event
            for event in first
            if event.event_type
            == MonitoringEventType.APPLICATION_REQUEST
        ]
    ) == DEFAULT_REQUEST_COUNT
    assert event_types == set(MonitoringEventType)
    assert {
        event.retrieval_strategy
        for event in first
        if event.retrieval_strategy
    } == {"semantic", "keyword", "hybrid"}
    assert {
        event.prompt_strategy for event in first if event.prompt_strategy
    } == {
        "baseline-concise",
        "grounded-evidence-first",
        "structured-troubleshooting",
    }
    assert {
        event.response_mode for event in first if event.response_mode
    } == {"concise", "detailed"}
    assert {
        (event.client_id, event.environment) for event in first
    } == {("demo-client-a", "dev"), ("demo-client-b", "test")}
    assert {
        event.client_id
        for event in first
        if event.event_type == MonitoringEventType.USER_FEEDBACK
    } == {"demo-client-a", "demo-client-b"}
    assert {
        "successful_grounded_response",
        "incomplete_citations",
        "no_result",
        "safety_sensitive",
        "approval_required",
        "application_error",
        "higher_cost",
        "slow_request",
    }.issubset(flags)
    assert any(
        event.event_type == MonitoringEventType.LLM_COMPLETION
        and (event.retrieved_source_count or 0) > (event.citation_count or 0)
        for event in first
    )
    assert any(
        event.event_type == MonitoringEventType.RETRIEVAL_COMPLETION
        and event.retrieved_source_count == 0
        for event in first
    )
    assert any(
        event.estimated_cost_usd is not None
        and event.estimated_cost_usd >= Decimal("0.001")
        for event in first
    )
    assert any(
        event.latency_ms is not None and event.latency_ms >= 1000
        for event in first
    )


def test_synthetic_fixture_requires_explicit_overwrite(tmp_path):
    path = tmp_path / "synthetic.jsonl"

    write_synthetic_monitoring_fixture(path)

    with pytest.raises(FileExistsError):
        write_synthetic_monitoring_fixture(path)
    write_synthetic_monitoring_fixture(path, overwrite=True)
    assert len(JsonLinesEventSink(path).load().events) == 275


def test_monitoring_analysis_produces_expected_review_metrics():
    analysis = _analysis()
    overview = analysis.overview

    assert overview["event_count"] == 275
    assert overview["request_count"] == 84
    assert overview["llm_completion_count"] == 62
    assert overview["request_success_rate"] == pytest.approx(80 / 84)
    assert overview["feedback_coverage_rate"] == 0.5
    assert overview["feedback_rate"] == 0.5
    assert overview["positive_feedback_rate"] == pytest.approx(31 / 42)
    assert overview["negative_feedback_rate"] == pytest.approx(11 / 42)
    assert overview["average_rating"] == pytest.approx(31 / 42)
    assert overview["safety_event_count"] == 9
    assert overview["approval_required_count"] == 9
    assert overview["p95_latency_ms"] > overview["p50_latency_ms"]
    assert overview["total_estimated_cost_usd"] == Decimal("0.02452375")
    assert set(analysis.by_retrieval_strategy) == {
        "semantic",
        "keyword",
        "hybrid",
    }
    assert set(analysis.by_prompt_strategy) == {
        "baseline-concise",
        "grounded-evidence-first",
        "structured-troubleshooting",
    }
    assert set(analysis.by_response_mode) == {"concise", "detailed"}
    assert set(analysis.by_runtime_mode) == {"offline-synthetic"}
    assert set(analysis.by_safety_outcome) == {
        "allowed",
        "approval_required",
        "blocked",
    }
    assert set(analysis.by_client) == {
        "demo-client-a",
        "demo-client-b",
    }
    assert set(analysis.by_environment) == {"dev", "test"}
    assert len(analysis.by_scope) == 2
    assert len(analysis.daily) >= 14
    assert analysis.metadata["data_classification"] == "synthetic"
    assert analysis.metadata["network_calls"] is False
    assert analysis.metadata["estimated_cost_is_simulated"] is True
    assert analysis.metadata["provider_charges_incurred"] is False


@pytest.mark.parametrize(
    "config",
    [
        lambda: MonitoringAnalysisConfig(
            high_cost_threshold_usd=Decimal("-1")
        ),
        lambda: MonitoringAnalysisConfig(
            high_cost_threshold_usd=0.1
        ),
        lambda: MonitoringAnalysisConfig(
            slow_request_threshold_ms=0
        ),
    ],
)
def test_monitoring_analysis_config_rejects_invalid_thresholds(config):
    with pytest.raises(ValueError):
        config()


def test_monitoring_analysis_rejects_empty_and_duplicate_events():
    event = _request_event()

    with pytest.raises(ValueError, match="At least one"):
        analyze_monitoring_events(())
    with pytest.raises(ValueError, match="unique"):
        analyze_monitoring_events((event, event))


def test_monitoring_analysis_never_joins_feedback_across_scopes():
    request_a = _request_event(prompt_strategy="baseline-concise")
    completion_a = replace(
        request_a,
        event_id="event-002",
        event_type=MonitoringEventType.LLM_COMPLETION,
        llm_provider="fake",
        model_id="model",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        retrieved_source_count=1,
        citation_count=1,
    )
    request_b = replace(
        request_a,
        event_id="event-003",
        client_id="client-b",
        environment="test",
        prompt_strategy="grounded-evidence-first",
    )
    completion_b = replace(
        request_b,
        event_id="event-004",
        event_type=MonitoringEventType.LLM_COMPLETION,
        llm_provider="fake",
        model_id="model",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        retrieved_source_count=1,
        citation_count=1,
    )
    feedback_b = replace(
        request_b,
        event_id="event-005",
        event_type=MonitoringEventType.USER_FEEDBACK,
        user_rating=UserRating.DOWN,
    )

    analysis = analyze_monitoring_events(
        (
            request_a,
            completion_a,
            request_b,
            completion_b,
            feedback_b,
        ),
        evaluated_at=FIXED_TIME_TEXT,
        git_commit=FIXED_COMMIT,
    )

    assert analysis.by_prompt_strategy["baseline-concise"][
        "feedback_count"
    ] == 0
    assert analysis.by_prompt_strategy["grounded-evidence-first"][
        "feedback_count"
    ] == 1


def test_report_generation_writes_exact_artifacts_and_pngs(tmp_path):
    analysis = _analysis()

    paths = write_monitoring_reports(analysis, tmp_path)
    payload = json.loads(
        (tmp_path / JSON_FILENAME).read_text(encoding="utf-8")
    )
    markdown = (tmp_path / MARKDOWN_FILENAME).read_text(
        encoding="utf-8"
    )
    strategy_csv = (tmp_path / STRATEGY_CSV_FILENAME).read_text(
        encoding="utf-8"
    )
    intent_csv = (tmp_path / INTENT_CSV_FILENAME).read_text(
        encoding="utf-8"
    )
    day_csv = (tmp_path / DAY_CSV_FILENAME).read_text(
        encoding="utf-8"
    )

    assert payload["overview"]["event_count"] == 275
    assert "**synthetic**" in markdown
    assert "Synthetic feedback:" not in markdown
    assert strategy_csv.startswith("strategy_type,strategy,")
    assert intent_csv.startswith("intent,request_count,")
    assert day_csv.startswith("date,event_count,")
    assert len(paths.chart_paths) == 6
    assert tuple(path.name for path in paths.chart_paths) == CHART_FILENAMES
    assert all(
        path.parent.name == CHART_DIRECTORY
        and path.is_file()
        and path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        for path in paths.chart_paths
    )
    assert {
        struct.unpack(">II", path.read_bytes()[16:24])
        for path in paths.chart_paths
    } == {(1200, 720)}
    assert render_monitoring_markdown(analysis) == markdown
    assert render_strategy_csv(analysis) == strategy_csv
    assert render_intent_csv(analysis) == intent_csv
    assert render_day_csv(analysis) == day_csv


def test_report_generation_is_deterministic(tmp_path):
    analysis = _analysis()
    first = write_monitoring_reports(analysis, tmp_path / "first")
    second = write_monitoring_reports(analysis, tmp_path / "second")

    assert first.json_path.read_bytes() == second.json_path.read_bytes()
    assert first.markdown_path.read_bytes() == (
        second.markdown_path.read_bytes()
    )
    assert first.strategy_csv_path.read_bytes() == (
        second.strategy_csv_path.read_bytes()
    )
    assert first.intent_csv_path.read_bytes() == (
        second.intent_csv_path.read_bytes()
    )
    assert first.day_csv_path.read_bytes() == second.day_csv_path.read_bytes()
    assert [
        path.read_bytes() for path in first.chart_paths
    ] == [path.read_bytes() for path in second.chart_paths]


def test_generator_cli_verifies_matching_fixture_and_protects_changes(
    tmp_path,
):
    path = tmp_path / "synthetic.jsonl"

    assert generate_main(["--output", str(path)]) == 0
    assert generate_main(["--output", str(path)]) == 0
    existing = JsonLinesEventSink(path).load().events[0]
    JsonLinesEventSink(path).append(existing)
    assert generate_main(["--output", str(path)]) == 2
    assert generate_main(["--output", str(path), "--force"]) == 0


def test_analysis_runner_rejects_malformed_input(tmp_path):
    path = tmp_path / "malformed.jsonl"
    path.write_text("not-json\n", encoding="utf-8")

    assert analysis_main(
        [
            "--input",
            str(path),
            "--output-dir",
            str(tmp_path / "results"),
        ]
    ) == 2


def test_analysis_runner_operates_with_network_sockets_disabled(
    tmp_path,
    monkeypatch,
):
    fixture = tmp_path / "synthetic.jsonl"
    write_synthetic_monitoring_fixture(fixture)

    def blocked_socket(*args, **kwargs):
        raise AssertionError("Network access is forbidden")

    monkeypatch.setattr(socket, "socket", blocked_socket)

    result = analysis_main(
        [
            "--input",
            str(fixture),
            "--output-dir",
            str(tmp_path / "results"),
            "--evaluated-at",
            FIXED_TIME_TEXT,
            "--git-commit",
            FIXED_COMMIT,
        ]
    )

    assert result == 0
    assert (tmp_path / "results" / JSON_FILENAME).is_file()


def test_dashboard_loads_only_redacted_synthetic_evidence(tmp_path):
    fixture = tmp_path / "monitoring_events.jsonl"
    results = tmp_path / "results"
    write_synthetic_monitoring_fixture(fixture)
    reports = write_monitoring_reports(_analysis(), results)

    dashboard = load_monitoring_dashboard_data(
        reports.json_path,
        fixture,
        recent_limit=10,
    )

    assert dashboard.overview["request_count"] == 84
    assert len(dashboard.retrieval_comparison) == 3
    assert len(dashboard.prompt_comparison) == 3
    assert len(dashboard.recent_events) == 10
    assert all(
        "feedback_text" not in row
        and "evaluation_metadata" not in row
        for row in dashboard.recent_events
    )
    assert '"data_classification": "synthetic"' in (
        dashboard.summary_download
    )


def test_dashboard_rejects_non_synthetic_summary(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "metadata": {"data_classification": "production"},
                "overview": {},
                "by_retrieval_strategy": {},
                "by_prompt_strategy": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="synthetic data only"):
        load_monitoring_dashboard_data(
            summary,
            tmp_path / "not-read.jsonl",
        )


def test_streamlit_monitoring_page_is_labeled_and_offline():
    app = AppTest.from_file("ui/app.py", default_timeout=20)

    app.run()
    app.radio[0].set_value("Offline monitoring").run()

    assert not app.error
    assert [metric.label for metric in app.metric] == [
        "Synthetic requests",
        "Success rate",
        "P95 latency",
        "Simulated total cost",
        "Positive feedback",
        "No-result rate",
    ]
    assert any(
        "Synthetic demonstration data only" in warning.value
        and "does not connect to AWS" in warning.value
        for warning in app.warning
    )
    assert len(app.dataframe) == 3
    assert len(app.get("download_button")) == 1


def test_gitignore_excludes_local_events_and_keeps_reviewed_fixture():
    ignore = Path(".gitignore").read_text(encoding="utf-8")
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")

    assert "data/monitoring/*.jsonl" in ignore
    assert "data/monitoring/**/*.jsonl" in ignore
    assert "!data/monitoring/synthetic_events.jsonl" not in ignore
    assert "evaluation/fixtures/monitoring_events.jsonl" not in ignore
    assert "data/monitoring" in dockerignore
