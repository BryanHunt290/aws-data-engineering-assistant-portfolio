"""Reviewer-friendly offline monitoring reports and Matplotlib charts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from evaluation.monitoring_analysis import MonitoringAnalysis


JSON_FILENAME = "monitoring_summary.json"
MARKDOWN_FILENAME = "monitoring_summary.md"
STRATEGY_CSV_FILENAME = "monitoring_by_strategy.csv"
INTENT_CSV_FILENAME = "monitoring_by_intent.csv"
DAY_CSV_FILENAME = "monitoring_by_day.csv"
CHART_DIRECTORY = "monitoring"
CHART_FILENAMES = (
    "request_volume.png",
    "latency_by_strategy.png",
    "cost_by_strategy.png",
    "feedback_summary.png",
    "error_rate.png",
    "token_usage.png",
)
_FIGURE_SIZE = (10, 6)
_DPI = 120


@dataclass(frozen=True)
class MonitoringReportPaths:
    """Paths created for one analysis run."""

    json_path: Path
    markdown_path: Path
    strategy_csv_path: Path
    intent_csv_path: Path
    day_csv_path: Path
    chart_paths: tuple[Path, ...]


def write_monitoring_reports(
    analysis: MonitoringAnalysis,
    output_directory: Path | str,
) -> MonitoringReportPaths:
    """Write the exact reviewed monitoring evidence artifact set."""

    destination = Path(output_directory)
    chart_directory = destination / CHART_DIRECTORY
    destination.mkdir(parents=True, exist_ok=True)
    chart_directory.mkdir(parents=True, exist_ok=True)

    json_path = destination / JSON_FILENAME
    markdown_path = destination / MARKDOWN_FILENAME
    strategy_csv_path = destination / STRATEGY_CSV_FILENAME
    intent_csv_path = destination / INTENT_CSV_FILENAME
    day_csv_path = destination / DAY_CSV_FILENAME

    json_path.write_text(
        json.dumps(analysis.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_monitoring_markdown(analysis),
        encoding="utf-8",
    )
    strategy_csv_path.write_text(
        render_strategy_csv(analysis),
        encoding="utf-8",
        newline="",
    )
    intent_csv_path.write_text(
        render_intent_csv(analysis),
        encoding="utf-8",
        newline="",
    )
    day_csv_path.write_text(
        render_day_csv(analysis),
        encoding="utf-8",
        newline="",
    )
    chart_paths = _write_charts(analysis, chart_directory)
    return MonitoringReportPaths(
        json_path=json_path,
        markdown_path=markdown_path,
        strategy_csv_path=strategy_csv_path,
        intent_csv_path=intent_csv_path,
        day_csv_path=day_csv_path,
        chart_paths=chart_paths,
    )


def render_monitoring_markdown(analysis: MonitoringAnalysis) -> str:
    """Render a compact reviewer-facing summary without raw event content."""

    metadata = analysis.metadata
    overview = analysis.overview
    lines = [
        "# Offline monitoring and feedback analysis",
        "",
        "All values in this report come from deterministic **synthetic** "
        "offline events. They are demonstration evidence, not production "
        "telemetry. No AWS services were called and no provider charges were "
        "incurred.",
        "",
        "## Provenance",
        "",
        f"- Evaluation date: `{metadata['evaluation_date']}`",
        f"- Git commit at generation: `{metadata['git_commit']}`",
        f"- Analysis version: `{metadata['analysis_version']}`",
        f"- Dataset versions: `{', '.join(metadata['dataset_versions'])}`",
        f"- Event schema version: `{metadata['event_schema_version']}`",
        f"- Python: `{metadata['python_version']}`",
        "- Data classification: **synthetic**",
        "- Estimated costs are simulated: `true`",
        "- Raw prompts, source documents, credentials, vectors, and raw "
        "feedback text are excluded.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Requests | {overview['request_count']} |",
        f"| Success rate | {_percent(overview['request_success_rate'])} |",
        f"| Error rate | {_percent(overview['application_error_rate'])} |",
        f"| Feedback rate | {_percent(overview['feedback_rate'])} |",
        f"| Positive feedback | "
        f"{_percent(overview['positive_feedback_rate'])} |",
        f"| Negative feedback | "
        f"{_percent(overview['negative_feedback_rate'])} |",
        f"| Average rating (0–1) | {overview['average_rating']:.3f} |",
        f"| Average latency | {overview['average_latency_ms']:.1f} ms |",
        f"| P50 latency | {overview['p50_latency_ms']:.1f} ms |",
        f"| P95 latency | {overview['p95_latency_ms']:.1f} ms |",
        f"| Average retrieval latency | "
        f"{overview['average_retrieval_latency_ms']:.1f} ms |",
        f"| Average generation latency | "
        f"{overview['average_generation_latency_ms']:.1f} ms |",
        f"| Average input tokens | "
        f"{overview['average_input_tokens']:.1f} |",
        f"| Average output tokens | "
        f"{overview['average_output_tokens']:.1f} |",
        f"| Average total tokens | "
        f"{overview['average_total_tokens']:.1f} |",
        f"| Simulated total estimated cost | "
        f"USD {overview['total_estimated_cost_usd']} |",
        f"| Simulated average cost per request | "
        f"USD {overview['average_estimated_cost_per_request_usd']} |",
        f"| Retrieval no-result rate | "
        f"{_percent(overview['no_result_rate'])} |",
        f"| Citation completion proxy | "
        f"{_percent(overview['citation_completeness_rate'])} |",
        f"| Safety events | {overview['safety_event_count']} |",
        f"| Approval-required events | "
        f"{overview['approval_required_count']} |",
        "",
        "## Retrieval strategy comparison",
        "",
        "| Strategy | Completions | Success | No result | Avg retrieval ms | P95 retrieval ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy, metric in analysis.by_retrieval_strategy.items():
        lines.append(
            f"| {strategy} | {metric['completion_count']} "
            f"| {_percent(metric['success_rate'])} "
            f"| {_percent(metric['no_result_rate'])} "
            f"| {metric['latency_ms']['average']:.1f} "
            f"| {metric['latency_ms']['p95']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Prompt strategy comparison",
            "",
            "| Strategy | Completions | Complete citations | Positive feedback | Avg tokens | Simulated avg USD | P95 latency ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for strategy, metric in analysis.by_prompt_strategy.items():
        lines.append(
            f"| {strategy} | {metric['completion_count']} "
            f"| {_percent(metric['citation_completeness_rate'])} "
            f"| {_percent(metric['positive_feedback_rate'])} "
            f"| {metric['average_total_tokens']:.1f} "
            f"| {metric['average_estimated_cost_usd']} "
            f"| {metric['latency_ms']['p95']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Charts",
            "",
            "![Synthetic request volume](monitoring/request_volume.png)",
            "",
            "![Synthetic latency by retrieval strategy](monitoring/latency_by_strategy.png)",
            "",
            "![Synthetic cost by prompt strategy](monitoring/cost_by_strategy.png)",
            "",
            "![Synthetic feedback summary](monitoring/feedback_summary.png)",
            "",
            "![Synthetic error rate](monitoring/error_rate.png)",
            "",
            "![Synthetic token usage](monitoring/token_usage.png)",
            "",
            "## Interpretation boundary",
            "",
            "The fixture deliberately mixes positive and negative scenarios "
            "to demonstrate aggregation and grouping. Differences are "
            "illustrative, not statistically significant and not evidence of "
            "real-user behavior or production service levels. A rating of "
            "`up` is represented as 1 and `down` as 0 for the average-rating "
            "metric.",
            "",
        ]
    )
    return "\n".join(lines)


def render_strategy_csv(analysis: MonitoringAnalysis) -> str:
    """Render retrieval and prompt strategy comparisons as a wide CSV."""

    fields = (
        "strategy_type",
        "strategy",
        "completion_count",
        "success_rate",
        "no_result_rate",
        "citation_completeness_rate",
        "positive_feedback_rate",
        "average_latency_ms",
        "p95_latency_ms",
        "average_total_tokens",
        "average_estimated_cost_usd",
    )
    rows: list[dict[str, Any]] = []
    for strategy, metric in analysis.by_retrieval_strategy.items():
        rows.append(
            {
                "strategy_type": "retrieval",
                "strategy": strategy,
                "completion_count": metric["completion_count"],
                "success_rate": metric["success_rate"],
                "no_result_rate": metric["no_result_rate"],
                "citation_completeness_rate": "",
                "positive_feedback_rate": "",
                "average_latency_ms": metric["latency_ms"]["average"],
                "p95_latency_ms": metric["latency_ms"]["p95"],
                "average_total_tokens": metric["average_total_tokens"],
                "average_estimated_cost_usd": metric[
                    "average_estimated_cost_usd"
                ],
            }
        )
    for strategy, metric in analysis.by_prompt_strategy.items():
        rows.append(
            {
                "strategy_type": "prompt",
                "strategy": strategy,
                "completion_count": metric["completion_count"],
                "success_rate": "",
                "no_result_rate": "",
                "citation_completeness_rate": metric[
                    "citation_completeness_rate"
                ],
                "positive_feedback_rate": metric[
                    "positive_feedback_rate"
                ],
                "average_latency_ms": metric["latency_ms"]["average"],
                "p95_latency_ms": metric["latency_ms"]["p95"],
                "average_total_tokens": metric["average_total_tokens"],
                "average_estimated_cost_usd": metric[
                    "average_estimated_cost_usd"
                ],
            }
        )
    return _render_csv(fields, rows)


def render_intent_csv(analysis: MonitoringAnalysis) -> str:
    """Render deterministic metrics grouped by intent."""

    fields = (
        "intent",
        "request_count",
        "application_error_rate",
        "feedback_count",
        "positive_feedback_rate",
    )
    return _render_csv(
        fields,
        (
            {"intent": intent, **metric}
            for intent, metric in analysis.by_intent.items()
        ),
    )


def render_day_csv(analysis: MonitoringAnalysis) -> str:
    """Render deterministic UTC-day aggregates."""

    fields = (
        "date",
        "event_count",
        "request_count",
        "llm_completion_count",
        "error_count",
        "feedback_count",
        "positive_feedback_count",
        "estimated_cost_usd",
    )
    return _render_csv(fields, analysis.daily)


def _write_charts(
    analysis: MonitoringAnalysis,
    directory: Path,
) -> tuple[Path, ...]:
    """Create six separate, stable-size PNG charts using default styling."""

    overview = analysis.overview
    retrieval = analysis.by_retrieval_strategy
    prompt = analysis.by_prompt_strategy

    paths = tuple(directory / name for name in CHART_FILENAMES)

    _new_figure()
    dates = [row["date"] for row in analysis.daily]
    plt.plot(dates, [row["request_count"] for row in analysis.daily], label="Requests")
    plt.plot(dates, [row["feedback_count"] for row in analysis.daily], label="Feedback")
    plt.title("Synthetic Offline Request Volume by UTC Day")
    plt.xlabel("UTC date")
    plt.ylabel("Event count")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    _save(paths[0])

    _new_figure()
    retrieval_names = list(retrieval)
    positions = list(range(len(retrieval_names)))
    width = 0.35
    plt.bar(
        [position - width / 2 for position in positions],
        [retrieval[name]["latency_ms"]["p50"] for name in retrieval_names],
        width,
        label="P50",
    )
    plt.bar(
        [position + width / 2 for position in positions],
        [retrieval[name]["latency_ms"]["p95"] for name in retrieval_names],
        width,
        label="P95",
    )
    plt.xticks(positions, retrieval_names)
    plt.title("Synthetic Retrieval Latency by Strategy")
    plt.xlabel("Retrieval strategy")
    plt.ylabel("Latency (milliseconds)")
    plt.legend()
    _save(paths[1])

    _new_figure()
    prompt_names = list(prompt)
    plt.bar(
        prompt_names,
        [
            float(prompt[name]["average_estimated_cost_usd"])
            for name in prompt_names
        ],
    )
    plt.title("Synthetic Average Estimated Cost by Prompt Strategy")
    plt.xlabel("Prompt strategy")
    plt.ylabel("Estimated cost per completion (USD)")
    plt.xticks(rotation=15, ha="right")
    _save(paths[2])

    _new_figure()
    feedback_labels = ("Positive", "Negative", "No feedback")
    feedback_values = (
        overview["positive_feedback_rate"],
        overview["negative_feedback_rate"],
        1.0 - overview["feedback_rate"],
    )
    plt.bar(feedback_labels, feedback_values)
    plt.title("Synthetic Feedback Summary")
    plt.xlabel("Feedback outcome")
    plt.ylabel("Rate")
    _save(paths[3])

    _new_figure()
    intent_names = list(analysis.by_intent)
    plt.bar(
        intent_names,
        [
            analysis.by_intent[name]["application_error_rate"]
            for name in intent_names
        ],
    )
    plt.title("Synthetic Application Error Rate by Intent")
    plt.xlabel("Intent")
    plt.ylabel("Error rate")
    plt.xticks(rotation=35, ha="right")
    _save(paths[4])

    _new_figure()
    positions = list(range(len(prompt_names)))
    plt.bar(
        [position - width / 2 for position in positions],
        [prompt[name]["average_input_tokens"] for name in prompt_names],
        width,
        label="Input tokens",
    )
    plt.bar(
        [position + width / 2 for position in positions],
        [prompt[name]["average_output_tokens"] for name in prompt_names],
        width,
        label="Output tokens",
    )
    plt.xticks(positions, prompt_names, rotation=15, ha="right")
    plt.title("Synthetic Average Token Usage by Prompt Strategy")
    plt.xlabel("Prompt strategy")
    plt.ylabel("Average tokens per completion")
    plt.legend()
    _save(paths[5])

    return paths


def _new_figure() -> None:
    plt.close("all")
    plt.figure(figsize=_FIGURE_SIZE, dpi=_DPI)


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=_DPI,
        metadata={
            "Software": "AWS Data Engineering Assistant offline evaluation",
            "Title": "Synthetic offline monitoring evidence",
        },
    )
    plt.close()


def _render_csv(
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]] | Any,
) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: _csv_safe(row.get(key, ""))
                for key in fieldnames
            }
        )
    return output.getvalue()


def _csv_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _percent(value: float) -> str:
    return f"{float(value) * 100:.1f}%"
