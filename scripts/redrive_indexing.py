"""Inspect/redrive S3 indexing descriptors; dry-run is the default."""

import argparse
from dataclasses import asdict
import json

from knowledge.indexing_redrive import (
    IndexingRedriveService,
    RedriveCandidate,
    RedriveFilters,
)
from knowledge.storage import S3KnowledgeStorage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--namespace")
    parser.add_argument("--domain")
    parser.add_argument("--document-id")
    parser.add_argument("--status")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--reset-retryable", action="store_true")
    parser.add_argument("--function-name")
    args = parser.parse_args()
    if args.apply and not args.function_name:
        parser.error("--apply requires --function-name")

    import boto3

    storage = S3KnowledgeStorage(args.bucket, boto3.client("s3"))
    service = IndexingRedriveService(storage)
    lambda_client = boto3.client("lambda") if args.apply else None

    def dispatch(candidate: RedriveCandidate) -> None:
        assert lambda_client is not None
        event = {
            "Records": [{
                "eventName": "ObjectCreated:Redrive",
                "s3": {
                    "bucket": {"name": args.bucket},
                    "object": {"key": candidate.raw_key, "size": 0},
                },
            }]
        }
        lambda_client.invoke(
            FunctionName=args.function_name,
            InvocationType="Event",
            Payload=json.dumps(event).encode("utf-8"),
        )

    report = service.redrive(
        RedriveFilters(
            client_id=args.client_id,
            environment=args.environment,
            namespace=args.namespace,
            domain=args.domain,
            document_id=args.document_id,
            status=args.status,
        ),
        apply=args.apply,
        reset_retryable=args.reset_retryable,
        dispatcher=dispatch if args.apply else None,
    )
    print(json.dumps(asdict(report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
