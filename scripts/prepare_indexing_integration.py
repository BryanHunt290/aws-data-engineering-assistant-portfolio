"""Prepare internal-dev indexing integration artifacts without network calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge.integration_preparation import (
    DEFAULT_AUDIT_PATH,
    DEFAULT_CONTEXT_PATH,
    DEFAULT_EXAMPLE_PATH,
    DEFAULT_EXPECTATIONS_PATH,
    DEFAULT_LOCAL_CONFIG_PATH,
    OPTIONAL_VPC_FIELDS,
    OPTIONAL_NON_SECRET_FIELDS,
    PROHIBITED_PLAINTEXT_FIELDS,
    REQUIRED_NON_SECRET_FIELDS,
    SECRET_REFERENCE_FIELDS,
    bootstrap_local_configuration,
    generate_context_artifacts,
    load_json_object,
    review_configuration,
    write_expected_resource_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline preparation for the controlled internal-dev production "
            "indexing integration test"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser(
        "bootstrap", description="Create the ignored local configuration"
    )
    bootstrap.add_argument("--source", type=Path, default=DEFAULT_EXAMPLE_PATH)
    bootstrap.add_argument("--output", type=Path, default=DEFAULT_LOCAL_CONFIG_PATH)
    bootstrap.add_argument("--force", action="store_true")

    review = subparsers.add_parser(
        "review", description="Review every configuration field offline"
    )
    review.add_argument("--config", required=True, type=Path)
    review.add_argument("--format", choices=("human", "json"), default="human")

    context = subparsers.add_parser(
        "generate-context",
        description="Generate reviewed PowerShell CDK context and audit artifacts",
    )
    context.add_argument("--config", required=True, type=Path)
    context.add_argument("--output", type=Path, default=DEFAULT_CONTEXT_PATH)
    context.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_PATH)
    context.add_argument("--force", action="store_true")

    expected = subparsers.add_parser(
        "expected-resources",
        description="Write deterministic expected-resource and template review",
    )
    expected.add_argument("--config", required=True, type=Path)
    expected.add_argument("--template", type=Path)
    expected.add_argument("--output", type=Path, default=DEFAULT_EXPECTATIONS_PATH)
    expected.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "bootstrap":
            output = bootstrap_local_configuration(
                source=args.source,
                output=args.output,
                force=args.force,
            )
            print(f"created: {output}")
            print("required non-secret identifiers: " + ", ".join(REQUIRED_NON_SECRET_FIELDS))
            print("optional VPC identifiers: " + ", ".join(OPTIONAL_VPC_FIELDS))
            print("secret references (ARNs only): " + ", ".join(SECRET_REFERENCE_FIELDS))
            print("optional non-secret references: " + ", ".join(OPTIONAL_NON_SECRET_FIELDS))
            print("prohibited plaintext secret fields: " + ", ".join(PROHIBITED_PLAINTEXT_FIELDS))
            print("No credential or secret value was populated or displayed.")
            return 0

        context = load_json_object(args.config)
        if args.command == "review":
            report = review_configuration(context)
            if args.format == "json":
                print(json.dumps(report.to_dict(), sort_keys=True))
            else:
                print(f"offline configuration review: {'READY' if report.ready else 'NOT READY'}")
                print("status                      category                    field")
                for field in report.fields:
                    print(f"{field.status.value:<27} {field.category:<27} {field.field}")
                print(
                    f"preflight: {'ready' if report.preflight.ready else 'not-ready'}; "
                    f"network calls: 0"
                )
            return 0 if report.ready else 2

        if args.command == "generate-context":
            context_path, audit_path, _ = generate_context_artifacts(
                context,
                config_path=args.config,
                context_output=args.output,
                audit_output=args.audit_output,
                force=args.force,
            )
            print(f"context artifact: {context_path}")
            print(f"audit artifact: {audit_path}")
            print("safe for: cdk synth --no-lookups")
            print("not approved for: cdk diff, cdk deploy")
            return 0

        template = load_json_object(args.template) if args.template else None
        output_path, report = write_expected_resource_report(
            context,
            output=args.output,
            template=template,
            force=args.force,
        )
        print(f"expected-resource report: {output_path}")
        template_result = report["template_review"]
        if args.template is not None and not template_result["valid"]:
            return 2
        return 0
    except (
        FileExistsError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
