"""CLI for deterministic offline LLM and prompt evaluation."""

import argparse
from pathlib import Path
import sys

from evaluation.prompt_benchmark import load_prompt_benchmark
from evaluation.prompt_comparison import (
    PromptComparisonConfig,
    run_prompt_comparison,
)
from evaluation.prompt_reporting import write_prompt_reports


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = (
    PROJECT_ROOT
    / "evaluation"
    / "benchmark"
    / "llm_prompt_benchmark.json"
)
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "evaluation" / "results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare prompt strategies and deterministic fake LLM modes "
            "without AWS or network access."
        )
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--latency-repetitions", type=int, default=5)
    parser.add_argument("--concise-word-limit", type=int, default=55)
    parser.add_argument("--detailed-word-limit", type=int, default=110)
    parser.add_argument("--evaluated-at")
    parser.add_argument(
        "--git-commit",
        help="Override generation commit for deterministic report tests.",
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    try:
        comparison = run_prompt_comparison(
            load_prompt_benchmark(args.benchmark),
            config=PromptComparisonConfig(
                latency_repetitions=args.latency_repetitions,
                concise_word_limit=args.concise_word_limit,
                detailed_word_limit=args.detailed_word_limit,
            ),
            evaluated_at=args.evaluated_at,
            git_commit=args.git_commit,
        )
        paths = write_prompt_reports(
            comparison,
            args.output_dir,
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"LLM prompt comparison failed: {error}", file=sys.stderr)
        return 2

    print(
        "LLM prompt comparison complete: "
        f"{comparison.metadata['benchmark_case_count']} cases, "
        f"{len(comparison.outcomes)} case/strategy/mode results"
    )
    for strategy_id, metric in comparison.metrics.items():
        print(
            f"- {strategy_id}: "
            f"quality={metric['overall_quality_score']:.4f}, "
            f"complete={metric['answer_completeness_rate']:.4f}, "
            f"avg_tokens={metric['average_total_tokens']:.1f}"
        )
    print(
        "Recommended strategy: "
        f"{comparison.selection['recommended_strategy']}"
    )
    for path in paths:
        try:
            display_path = path.relative_to(PROJECT_ROOT)
        except ValueError:
            display_path = path
        print(f"Wrote {display_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
