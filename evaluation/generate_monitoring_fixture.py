"""CLI for the deterministic reviewed synthetic monitoring fixture."""

import argparse
from pathlib import Path
import sys

from evaluation.monitoring_dataset import (
    DEFAULT_REQUEST_COUNT,
    DEFAULT_SYNTHETIC_EVENT_PATH,
    SYNTHETIC_RANDOM_SEED,
    generate_synthetic_monitoring_events,
    write_synthetic_monitoring_fixture,
)
from knowledge.monitoring import JsonLinesEventSink


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate privacy-safe synthetic JSONL monitoring events "
            "without network or AWS access."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SYNTHETIC_EVENT_PATH,
    )
    parser.add_argument(
        "--request-count",
        type=int,
        default=DEFAULT_REQUEST_COUNT,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=SYNTHETIC_RANDOM_SEED,
        help="Fixed seed used for reproducible synthetic values.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the exact output fixture if it already exists.",
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    try:
        if args.output.exists() and not args.force:
            existing = JsonLinesEventSink(args.output).load(
                skip_malformed=False
            )
            expected = generate_synthetic_monitoring_events(
                request_count=args.request_count,
                random_seed=args.random_seed,
            )
            if existing.events != expected:
                raise ValueError(
                    "Synthetic monitoring fixture exists but differs from "
                    "the requested deterministic fixture; use --force to "
                    "replace it"
                )
            print(
                f"Verified {len(existing.events)} synthetic monitoring "
                f"events in {args.output}"
            )
            return 0
        path = write_synthetic_monitoring_fixture(
            args.output,
            request_count=args.request_count,
            random_seed=args.random_seed,
            overwrite=args.force,
        )
        result = JsonLinesEventSink(path).load(skip_malformed=False)
    except (OSError, TypeError, ValueError) as error:
        print(
            f"Synthetic monitoring generation failed: {error}",
            file=sys.stderr,
        )
        return 2
    print(
        f"Wrote {len(result.events)} synthetic monitoring events to {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
