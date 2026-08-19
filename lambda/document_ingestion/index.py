"""AWS Lambda entry point for S3-backed knowledge ingestion."""

import logging
import os

import boto3

from knowledge.config import KnowledgeConfig
from knowledge.event_ingestion import S3DocumentIngestionProcessor
from knowledge.extraction import EXTRACTABLE_DOCUMENT_TYPES
from knowledge.ingestion import KnowledgeIngestionPipeline
from knowledge.indexing_configuration import (
    AutomaticIndexingConfig,
    build_automatic_indexing_workflow,
)
from knowledge.manifest import KnowledgeManifestRepository
from knowledge.storage import S3KnowledgeStorage


logger = logging.getLogger()
logger.setLevel(logging.INFO)

_processor: S3DocumentIngestionProcessor | None = None


def _positive_integer(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _non_negative_integer(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _supported_document_types() -> frozenset[str]:
    configured = os.environ.get(
        "KNOWLEDGE_SUPPORTED_DOCUMENT_TYPES",
        ",".join(sorted(EXTRACTABLE_DOCUMENT_TYPES)),
    )
    values = frozenset(
        value.strip().lower().lstrip(".")
        for value in configured.split(",")
        if value.strip().lstrip(".")
    )
    if not values:
        raise ValueError(
            "KNOWLEDGE_SUPPORTED_DOCUMENT_TYPES cannot be empty"
        )
    unsupported = values - EXTRACTABLE_DOCUMENT_TYPES
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(
            "No registered Lambda text extractor for: "
            f"{names}"
        )
    return values


def _build_processor() -> S3DocumentIngestionProcessor:
    bucket_name = os.environ["KNOWLEDGE_BUCKET_NAME"]
    raw_prefix = os.environ.get("KNOWLEDGE_RAW_PREFIX", "knowledge/raw/")
    config = KnowledgeConfig(
        chunk_size=_positive_integer("KNOWLEDGE_CHUNK_SIZE", 1_000),
        overlap=_non_negative_integer("KNOWLEDGE_CHUNK_OVERLAP", 100),
        supported_document_types=_supported_document_types(),
        maximum_upload_size=_positive_integer(
            "KNOWLEDGE_MAXIMUM_UPLOAD_SIZE",
            10 * 1024 * 1024,
        ),
    )
    s3_client = boto3.client("s3")
    storage = S3KnowledgeStorage(bucket_name, s3_client)
    indexing_config = AutomaticIndexingConfig.from_environment(os.environ)
    manifest = KnowledgeManifestRepository(
        storage,
        maximum_conflict_retries=(
            indexing_config.manifest_conflict_retry_limit
        ),
    )
    indexing_service = build_automatic_indexing_workflow(
        indexing_config,
        storage=storage,
        manifest=manifest,
    )
    pipeline = KnowledgeIngestionPipeline(
        storage,
        config,
        manifest=manifest,
        event_logger=logger,
    )
    return S3DocumentIngestionProcessor(
        bucket_name=bucket_name,
        raw_prefix=raw_prefix,
        s3_client=s3_client,
        storage=storage,
        pipeline=pipeline,
        manifest=manifest,
        config=config,
        client_id=os.environ["CLIENT_ID"],
        environment=os.environ["DEPLOYMENT_ENVIRONMENT"],
        indexing_service=indexing_service,
        knowledge_namespace=indexing_config.knowledge_namespace,
        knowledge_domain=indexing_config.knowledge_domain,
        event_logger=logger,
    )


def handler(event, context):
    """Process an S3 ObjectCreated event without invoking Bedrock."""

    del context
    global _processor
    if _processor is None:
        _processor = _build_processor()
    return _processor.process_event(event)
