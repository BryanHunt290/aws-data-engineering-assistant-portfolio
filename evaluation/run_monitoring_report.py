"""CLI for deterministic offline monitoring and feedback analysis."""

import argparse
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sys

from evaluation.monitoring_analysis import (
    MonitoringAnalysisConfig,
    analyze_monitoring_events,
)
from evaluation.monitoring_dataset import DEFAULT_SYNTHETIC_EVENT_PATH
from evaluation.monitoring_reporting import write_monitoring_reports
from knowledge.monitoring import JsonLinesEventSink


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "evaluation" / "results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze local JSONL monitoring events and create synthetic "
            "review evidence without network or AWS access."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_SYNTHETIC_EVENT_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--client")
    parser.add_argument("--environment")
    parser.add_argument(
        "--high-cost-threshold-usd",
        default="0.001",
    )
    parser.add_argument(
        "--slow-request-threshold-ms",
        type=float,
        default=1000.0,
    )
    parser.add_argument("--evaluated-at")
    parser.add_argument("--git-commit")
    parser.add_argument(
        "--skip-malformed",
        action="store_true",
        help="Continue while reporting malformed JSONL record counts.",
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    try:
        try:
            high_cost_threshold = Decimal(
                args.high_cost_threshold_usd
            )
        except (InvalidOperation, TypeError) as error:
            raise ValueError(
                "high-cost-threshold-usd must be a Decimal"
            ) from error
        loaded = JsonLinesEventSink(args.input).load(
            client_id=args.client,
            environment=args.environment,
            skip_malformed=args.skip_malformed,
        )
        analysis = analyze_monitoring_events(
            loaded.events,
            config=MonitoringAnalysisConfig(
                high_cost_threshold_usd=high_cost_threshold,
                slow_request_threshold_ms=(
                    args.slow_request_threshold_ms
                ),
            ),
            evaluated_at=args.evaluated_at,
            git_commit=args.git_commit,
            malformed_record_count=len(loaded.malformed_records),
        )
        reports = write_monitoring_reports(
            analysis,
            args.output_dir,
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"Monitoring analysis failed: {error}", file=sys.stderr)
        return 2

    overview = analysis.overview
    print(
        "Monitoring analysis complete: "
        f"{overview['event_count']} events, "
        f"{overview['request_count']} requests"
    )
    print(
        "- request_success="
        f"{overview['request_success_rate']:.4f}, "
        "positive_feedback="
        f"{overview['positive_feedback_rate']:.4f}, "
        "citation_complete="
        f"{overview['citation_completeness_rate']:.4f}"
    )
    print(f"Wrote {reports.json_path}")
    print(f"Wrote {reports.markdown_path}")
    print(f"Wrote {reports.strategy_csv_path}")
    print(f"Wrote {reports.intent_csv_path}")
    print(f"Wrote {reports.day_csv_path}")
    for chart in reports.chart_paths:
        print(f"Wrote {chart}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
