"""Run split-aware offline evaluations against the 36-document dataset."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from evaluation.aws_pipeline_operations_benchmark import (
    DEFAULT_DATASET_ROOT,
    VALID_SPLITS,
    AWSPipelineOperationsBenchmarks,
    load_aws_pipeline_operations_benchmarks,
)
from evaluation.comparison import ComparisonConfig, run_comparison
from evaluation.prompt_comparison import (
    PromptComparisonConfig,
    run_prompt_comparison,
)
from evaluation.prompt_reporting import write_prompt_reports
from evaluation.reporting import write_reports


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "evaluation" / "results" / "aws_pipeline_operations"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval strategies and deterministic prompt contracts "
            "against one leakage-safe AWS pipeline operations split."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--split",
        choices=sorted(VALID_SPLITS),
        default="test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument("--latency-repetitions", type=int, default=5)
    parser.add_argument("--evaluated-at")
    parser.add_argument("--git-commit")
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    try:
        benchmarks = load_aws_pipeline_operations_benchmarks(
            args.dataset_root,
            split=args.split,
        )
        retrieval = run_comparison(
            benchmarks.retrieval,
            config=ComparisonConfig(
                latency_repetitions=args.latency_repetitions
            ),
            corpus_directory=benchmarks.corpus_directory,
            evaluated_at=args.evaluated_at,
        )
        answers = run_prompt_comparison(
            benchmarks.answers,
            config=PromptComparisonConfig(
                latency_repetitions=args.latency_repetitions
            ),
            corpus_directory=benchmarks.corpus_directory,
            evaluated_at=args.evaluated_at,
            git_commit=args.git_commit,
        )
        write_reports(retrieval, args.output_dir / "retrieval")
        write_prompt_reports(answers, args.output_dir / "answers")
        summary_paths = write_summary(
            benchmarks,
            retrieval=retrieval.to_dict(),
            answers=answers.to_dict(),
            output_directory=args.output_dir,
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"AWS pipeline operations evaluation failed: {error}", file=sys.stderr)
        return 2

    print(
        "AWS pipeline operations evaluation complete: "
        f"{benchmarks.retrieval_query_count} retrieval queries "
        f"({len(benchmarks.unanswerable_query_ids)} unanswerable recorded), "
        f"{len(benchmarks.answers.cases)} answer cases"
    )
    print(
        "Recommended retrieval strategy: "
        f"{retrieval.selection['recommended_default']}"
    )
    print(
        "Recommended prompt strategy: "
        f"{answers.selection['recommended_strategy']}"
    )
    for path in summary_paths:
        print(f"Wrote {_display_path(path)}")
    return 0


def write_summary(
    benchmarks: AWSPipelineOperationsBenchmarks,
    *,
    retrieval: dict[str, Any],
    answers: dict[str, Any],
    output_directory: Path | str,
) -> tuple[Path, Path]:
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "dataset": {
            "name": "AWS Data Pipeline Operations Knowledge Base",
            "version": benchmarks.dataset_version,
            "split": benchmarks.split,
            "license": "CC-BY-4.0",
            "document_count": len(benchmarks.document_ids),
            "retrieval_query_count": benchmarks.retrieval_query_count,
            "answerable_retrieval_query_count": len(
                benchmarks.retrieval.cases
            ),
            "unanswerable_retrieval_query_count": len(
                benchmarks.unanswerable_query_ids
            ),
            "answer_case_count": len(benchmarks.answers.cases),
        },
        "retrieval": {
            "recommended_strategy": retrieval["selection"][
                "recommended_default"
            ],
            "metrics": retrieval["metrics"],
        },
        "answers": {
            "recommended_prompt_strategy": answers["selection"][
                "recommended_strategy"
            ],
            "metrics": answers["metrics"],
        },
        "limitations": [
            "Unanswerable retrieval cases are preserved and counted but are "
            "not included in precision, recall, or MRR calculations.",
            "Answer evaluation uses a deterministic fake provider to measure "
            "prompt-contract adherence, not real language-model quality.",
            "All documents and cases are synthetic; no provider call or "
            "deployment occurs.",
        ],
    }
    json_path = destination / "evaluation_summary.json"
    markdown_path = destination / "evaluation_summary.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_summary(payload), encoding="utf-8")
    return json_path, markdown_path


def _render_summary(payload: dict[str, Any]) -> str:
    dataset = payload["dataset"]
    retrieval = payload["retrieval"]
    answers = payload["answers"]
    lines = [
        "# AWS pipeline operations evaluation summary",
        "",
        "Offline, deterministic evidence for the leakage-safe dataset split.",
        "",
        "## Coverage",
        "",
        f"- Dataset version / split: `{dataset['version']}` / "
        f"`{dataset['split']}`",
        f"- Documents: `{dataset['document_count']}`",
        f"- Retrieval queries: `{dataset['retrieval_query_count']}` "
        f"(`{dataset['answerable_retrieval_query_count']}` scored, "
        f"`{dataset['unanswerable_retrieval_query_count']}` unanswerable)",
        f"- Answer cases: `{dataset['answer_case_count']}`",
        "",
        "## Selections",
        "",
        "- Recommended retrieval strategy: "
        f"`{retrieval['recommended_strategy']}`",
        "- Recommended prompt strategy: "
        f"`{answers['recommended_prompt_strategy']}`",
        "",
        "## Retrieval metrics",
        "",
        "| Strategy | MRR | Hit rate | Recall@5 | No-result rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for strategy in ("semantic", "keyword", "hybrid"):
        metrics = retrieval["metrics"][strategy]
        lines.append(
            f"| {strategy} | {metrics['mrr']:.4f} | "
            f"{metrics['hit_rate']:.4f} | {metrics['recall_at_5']:.4f} | "
            f"{metrics['no_result_rate']:.4f} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in payload["limitations"])
    return "\n".join(lines) + "\n"


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


if __name__ == "__main__":
    raise SystemExit(main())
