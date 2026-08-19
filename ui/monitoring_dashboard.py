"""Read-only Streamlit view over reviewed synthetic monitoring evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import streamlit as st

from knowledge.monitoring import JsonLinesEventSink


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / "monitoring_summary.json"
)
DEFAULT_FIXTURE_PATH = (
    PROJECT_ROOT / "evaluation" / "fixtures" / "monitoring_events.jsonl"
)
DEFAULT_CHART_DIRECTORY = (
    PROJECT_ROOT / "evaluation" / "results" / "monitoring"
)
_SAFE_RECENT_FIELDS = (
    "timestamp",
    "event_type",
    "client_id",
    "environment",
    "runtime_mode",
    "intent",
    "retrieval_strategy",
    "prompt_strategy",
    "success",
    "error_category",
    "user_rating",
)


@dataclass(frozen=True)
class MonitoringDashboardData:
    """Validated aggregate and redacted event values for the UI."""

    overview: dict[str, Any]
    retrieval_comparison: tuple[dict[str, Any], ...]
    prompt_comparison: tuple[dict[str, Any], ...]
    recent_events: tuple[dict[str, Any], ...]
    summary_download: str


def load_monitoring_dashboard_data(
    summary_path: Path | str = DEFAULT_SUMMARY_PATH,
    fixture_path: Path | str = DEFAULT_FIXTURE_PATH,
    *,
    recent_limit: int = 15,
) -> MonitoringDashboardData:
    """Load only synthetic reviewed artifacts and omit free-form text."""

    if (
        isinstance(recent_limit, bool)
        or not isinstance(recent_limit, int)
        or recent_limit < 1
        or recent_limit > 100
    ):
        raise ValueError("recent_limit must be an integer from 1 to 100")

    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("Monitoring summary must be a JSON object")
    metadata = _mapping(summary, "metadata")
    if metadata.get("data_classification") != "synthetic":
        raise ValueError("Monitoring dashboard accepts synthetic data only")
    overview = dict(_mapping(summary, "overview"))
    retrieval = _comparison_rows(
        _mapping(summary, "by_retrieval_strategy"),
        metric_names=("success_rate", "no_result_rate"),
    )
    prompt = _comparison_rows(
        _mapping(summary, "by_prompt_strategy"),
        metric_names=(
            "citation_completeness_rate",
            "positive_feedback_rate",
        ),
    )

    loaded = JsonLinesEventSink(fixture_path).load(skip_malformed=False)
    if not loaded.events:
        raise ValueError("Synthetic monitoring fixture is empty")
    if any(
        event.evaluation_metadata.get("synthetic") is not True
        for event in loaded.events
    ):
        raise ValueError("Monitoring dashboard accepts synthetic events only")
    recent = sorted(
        loaded.events,
        key=lambda event: (event.timestamp, event.event_id),
        reverse=True,
    )[:recent_limit]
    recent_rows = tuple(
        {
            field: event.to_dict().get(field)
            for field in _SAFE_RECENT_FIELDS
        }
        for event in recent
    )
    return MonitoringDashboardData(
        overview=overview,
        retrieval_comparison=retrieval,
        prompt_comparison=prompt,
        recent_events=recent_rows,
        summary_download=(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        ),
    )


@st.cache_data(show_spinner=False)
def _cached_dashboard_data() -> MonitoringDashboardData:
    return load_monitoring_dashboard_data()


def render_monitoring_dashboard() -> None:
    """Render the offline synthetic evidence page with native components."""

    st.header("Offline monitoring and feedback")
    st.warning(
        "Synthetic demonstration data only — this is not live production "
        "monitoring and does not connect to AWS."
    )
    try:
        data = _cached_dashboard_data()
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        st.error(
            "Reviewed monitoring evidence could not be loaded. Regenerate it "
            f"with `python -m evaluation.run_monitoring_report`. ({error})"
        )
        return

    overview = data.overview
    with st.container(horizontal=True):
        st.metric("Synthetic requests", overview["request_count"])
        st.metric(
            "Success rate",
            _percent(overview["request_success_rate"]),
        )
        st.metric(
            "P95 latency",
            f"{overview['p95_latency_ms']:.1f} ms",
        )
    with st.container(horizontal=True):
        st.metric(
            "Simulated total cost",
            f"${overview['total_estimated_cost_usd']}",
        )
        st.metric(
            "Positive feedback",
            _percent(overview["positive_feedback_rate"]),
        )
        st.metric(
            "No-result rate",
            _percent(overview["no_result_rate"]),
        )

    st.subheader("Retrieval strategy comparison")
    st.caption("Rates calculated from deterministic synthetic events.")
    st.image(
        str(DEFAULT_CHART_DIRECTORY / "latency_by_strategy.png"),
        caption="Synthetic retrieval latency comparison",
    )
    st.dataframe(
        data.retrieval_comparison,
        hide_index=True,
        width="stretch",
    )

    st.subheader("Prompt strategy comparison")
    st.caption("Synthetic citation and feedback proxies; no real LLM calls.")
    st.image(
        str(DEFAULT_CHART_DIRECTORY / "cost_by_strategy.png"),
        caption="Synthetic prompt cost comparison",
    )
    st.dataframe(
        data.prompt_comparison,
        hide_index=True,
        width="stretch",
    )

    st.subheader("Recent synthetic events")
    st.caption(
        "Free-form feedback, prompts, document content, credentials, and "
        "vectors are intentionally excluded."
    )
    st.dataframe(
        data.recent_events,
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "Download synthetic monitoring summary",
        data=data.summary_download,
        file_name="monitoring_summary.json",
        mime="application/json",
    )


def _mapping(
    value: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"Monitoring summary is missing {key}")
    return result


def _comparison_rows(
    values: Mapping[str, Any],
    *,
    metric_names: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for strategy in sorted(values):
        metrics = values[strategy]
        if not isinstance(metrics, Mapping):
            raise ValueError("Monitoring strategy metrics must be objects")
        row = {"strategy": strategy}
        for metric_name in metric_names:
            metric = metrics.get(metric_name)
            if isinstance(metric, bool) or not isinstance(metric, (int, float)):
                raise ValueError(
                    f"Monitoring strategy is missing {metric_name}"
                )
            row[metric_name] = float(metric)
        rows.append(row)
    if not rows:
        raise ValueError("Monitoring strategy groups cannot be empty")
    return tuple(rows)


def _percent(value: float) -> str:
    return f"{float(value) * 100:.1f}%"
