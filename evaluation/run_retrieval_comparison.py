"""CLI entry point for the reproducible offline retrieval comparison."""

import argparse
from pathlib import Path
import sys

from evaluation.benchmark import load_benchmark
from evaluation.comparison import ComparisonConfig, run_comparison
from evaluation.reporting import write_reports


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = (
    PROJECT_ROOT / "evaluation" / "benchmark" / "retrieval_benchmark.json"
)
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "evaluation" / "results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare semantic, BM25 keyword, and reciprocal-rank hybrid "
            "retrieval using the synthetic offline corpus."
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
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--semantic-minimum-similarity",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--keyword-minimum-score",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--hybrid-minimum-score",
        type=float,
        default=0.0,
    )
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--semantic-weight", type=float, default=1.0)
    parser.add_argument("--keyword-weight", type=float, default=1.0)
    parser.add_argument("--rrf-rank-constant", type=int, default=60)
    parser.add_argument("--candidate-pool-size", type=int, default=50)
    parser.add_argument("--latency-repetitions", type=int, default=5)
    parser.add_argument(
        "--evaluated-at",
        help=(
            "Optional ISO timestamp for deterministic report tests; "
            "defaults to the current UTC time."
        ),
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
        benchmark = load_benchmark(args.benchmark)
        comparison = run_comparison(
            benchmark,
            config=ComparisonConfig(
                top_k=args.top_k,
                semantic_minimum_similarity=(
                    args.semantic_minimum_similarity
                ),
                keyword_minimum_score=args.keyword_minimum_score,
                hybrid_minimum_score=args.hybrid_minimum_score,
                bm25_k1=args.bm25_k1,
                bm25_b=args.bm25_b,
                semantic_weight=args.semantic_weight,
                keyword_weight=args.keyword_weight,
                rrf_rank_constant=args.rrf_rank_constant,
                candidate_pool_size=args.candidate_pool_size,
                latency_repetitions=args.latency_repetitions,
            ),
            evaluated_at=args.evaluated_at,
        )
        paths = write_reports(comparison, args.output_dir)
    except (OSError, TypeError, ValueError) as error:
        print(f"Retrieval comparison failed: {error}", file=sys.stderr)
        return 2

    metrics = comparison.metrics
    print(
        "Retrieval comparison complete: "
        f"{comparison.metadata['benchmark_case_count']} cases"
    )
    for strategy in ("semantic", "keyword", "hybrid"):
        values = metrics[strategy]
        print(
            f"- {strategy}: MRR={values['mrr']:.4f}, "
            f"hit@5={values['hit_rate']:.4f}, "
            f"p95={values['p95_latency_ms']:.3f} ms"
        )
    print(
        "Recommended default: "
        f"{comparison.selection['recommended_default']}"
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
