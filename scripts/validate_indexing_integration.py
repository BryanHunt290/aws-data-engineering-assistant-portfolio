"""Run offline integration preflight and produce a dry-run phase plan."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge.integration_validation import (
    IntegrationPreflightValidator,
    IntegrationValidationRunner,
)


def load_context(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Integration configuration must be a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate non-production indexing integration configuration "
            "without making network calls"
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        context = load_context(args.config)
        validator = IntegrationPreflightValidator()
        report = validator.validate(context)
        output: dict[str, object] = {"preflight": report.to_dict()}
        if report.ready and not args.preflight_only:
            output["phase_plan"] = IntegrationValidationRunner(
                preflight=validator
            ).plan(context).to_dict()
        print(json.dumps(output, sort_keys=True))
        return 0 if report.ready else 2
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        print(
            json.dumps(
                {
                    "preflight": {
                        "ready": False,
                        "error_count": 1,
                        "errors": ["configuration_file_invalid"],
                        "network_calls": 0,
                    }
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
