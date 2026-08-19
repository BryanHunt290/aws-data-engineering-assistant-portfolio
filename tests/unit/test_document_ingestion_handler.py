from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import Mock, patch
from urllib.parse import quote_plus

import pytest

from knowledge.config import KnowledgeConfig
from knowledge.event_ingestion import (
    AutomaticIndexingIncompleteError,
    IngestionBatchError,
    S3DocumentIngestionProcessor,
)
from knowledge.extraction import EXTRACTABLE_DOCUMENT_TYPES
from knowledge.ingestion import KnowledgeIngestionPipeline
from knowledge.manifest import KnowledgeManifestRepository
from knowledge.media_classification import (
    AUDIO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    MEDIA_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from knowledge.pdf_extraction import InvalidPdfError
from tests.unit.pdf_fixtures import make_text_pdf


FIXED_TIME = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)


class MemoryKnowledgeStorage:
    def __init__(self) -> None:
        self.byte_objects: dict[str, dict[str, Any]] = {}
        self.json_objects: dict[str, dict[str, Any]] = {}
        self.write_count = 0

    def put_bytes(
        self,
        key: str,
        content: bytes,
        *,
        content_type: str,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        self.write_count += 1
        self.byte_objects[key] = {
            "content": content,
            "content_type": content_type,
            "metadata": dict(metadata or {}),
        }

    def put_json(self, key: str, payload: Mapping[str, Any]) -> None:
        self.write_count += 1
        self.json_objects[key] = deepcopy(dict(payload))

    def get_json(self, key: str) -> dict[str, Any] | None:
        value = self.json_objects.get(key)
        return deepcopy(value) if value is not None else None


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail_keys: set[str] = set()
        self.get_calls: list[dict[str, str]] = []
        self.metadata: dict[str, dict[str, str]] = {}
        self.content_types: dict[str, str | None] = {}
        self.copy_calls: list[dict[str, object]] = []

    def add(
        self,
        key: str,
        content: bytes,
        *,
        metadata: dict[str, str] | None = None,
        content_type: str | None = None,
    ) -> None:
        self.objects[key] = content
        self.metadata[key] = dict(metadata or {})
        self.content_types[key] = content_type

    def get_object(self, **kwargs):
        self.get_calls.append(dict(kwargs))
        key = kwargs["Key"]
        if key in self.fail_keys:
            raise RuntimeError("simulated source read failure")
        return {
            "Body": BytesIO(self.objects[key]),
            "ETag": '"stable-etag"',
            "Metadata": self.metadata[key],
            **(
                {"ContentType": self.content_types[key]}
                if self.content_types[key] is not None
                else {}
            ),
            **(
                {"VersionId": kwargs["VersionId"]}
                if "VersionId" in kwargs
                else {}
            ),
        }

    def copy_object(self, **kwargs):
        self.copy_calls.append(dict(kwargs))
        source = kwargs["CopySource"]
        self.objects[kwargs["Key"]] = self.objects[source["Key"]]
        return {"CopyObjectResult": {"ETag": '"copied-etag"'}}


def _record(
    key: str,
    *,
    size: int = 5,
    version_id: str | None = None,
) -> dict:
    object_details: dict[str, object] = {
        "key": quote_plus(key, safe="/"),
        "size": size,
        "eTag": "stable-etag",
    }
    if version_id is not None:
        object_details["versionId"] = version_id
    return {
        "eventTime": "2026-07-30T12:00:00Z",
        "eventSource": "aws:s3",
        "eventName": "ObjectCreated:Put",
        "s3": {
            "bucket": {"name": "knowledge-bucket"},
            "object": object_details,
        },
    }


def _build_processor(*, indexing_service=None):
    storage = MemoryKnowledgeStorage()
    s3_client = FakeS3Client()
    config = KnowledgeConfig(
        chunk_size=8,
        overlap=2,
        supported_document_types=EXTRACTABLE_DOCUMENT_TYPES,
    )
    manifest = KnowledgeManifestRepository(storage)
    pipeline = KnowledgeIngestionPipeline(
        storage,
        config,
        manifest=manifest,
        clock=lambda: FIXED_TIME,
    )
    processor = S3DocumentIngestionProcessor(
        bucket_name="knowledge-bucket",
        raw_prefix="knowledge/raw/",
        s3_client=s3_client,
        storage=storage,
        pipeline=pipeline,
        manifest=manifest,
        config=config,
        client_id="internal",
        environment="dev",
        indexing_service=indexing_service,
        knowledge_namespace="runbooks",
        knowledge_domain="data-engineering",
        event_logger=logging.getLogger("test.document-ingestion"),
        clock=lambda: FIXED_TIME,
    )
    return processor, storage, s3_client


def _expected_summary(
    *,
    received: int,
    processed: int,
    skipped: int,
    failed: int,
    indexable: int = 0,
    media: int = 0,
    unsupported: int = 0,
    suspicious: int = 0,
    blocked: int = 0,
    mime_mismatch: int = 0,
) -> dict[str, int]:
    return {
        "records_received": received,
        "records_processed": processed,
        "records_skipped": skipped,
        "records_failed": failed,
        "indexable_documents_received": indexable,
        "media_objects_stored": media,
        "unsupported_binaries_stored": unsupported,
        "suspicious_objects_quarantined": suspicious,
        "media_indexing_attempts_blocked": blocked,
        "mime_mismatch_count": mime_mismatch,
    }


def _indexing_report(*, succeeded: bool):
    report = Mock()
    report.document_id = "event-document"
    report.succeeded = succeeded
    report.vector_status.value = "complete" if succeeded else "partial"
    report.statistics.failed_chunk_count = 0 if succeeded else 1
    return report


def test_one_valid_upload_processes_existing_source_and_returns_counters():
    processor, storage, s3_client = _build_processor()
    key = "knowledge/raw/guide.txt"
    s3_client.add(key, b"hello")

    result = processor.process_event({"Records": [_record(key)]})

    assert result == _expected_summary(
        received=1,
        processed=1,
        skipped=0,
        failed=0,
        indexable=1,
    )
    manifest = storage.json_objects["knowledge/metadata/manifest.json"]
    entry = next(iter(manifest["documents"].values()))
    assert entry["raw_key"] == key
    assert entry["chunk_count"] == 1
    assert entry["embedding_status"] == "pending"
    assert not any(
        object_key.startswith("knowledge/raw/")
        for object_key in storage.byte_objects
    )


def test_event_processor_automatically_indexes_with_injected_service():
    indexing_service = Mock()
    indexing_service.index_pending_document.return_value = _indexing_report(
        succeeded=True
    )
    processor, _, s3_client = _build_processor(
        indexing_service=indexing_service
    )
    key = "knowledge/raw/automatic.txt"
    s3_client.add(key, b"hello")

    result = processor.process_event({"Records": [_record(key)]})

    assert result["records_processed"] == 1
    _, kwargs = indexing_service.index_pending_document.call_args
    assert kwargs == {
        "client_id": "internal",
        "environment": "dev",
        "knowledge_namespace": "runbooks",
        "knowledge_domain": "data-engineering",
    }


def test_incomplete_automatic_indexing_uses_existing_event_retry_path():
    indexing_service = Mock()
    indexing_service.index_pending_document.side_effect = (
        _indexing_report(succeeded=False),
        _indexing_report(succeeded=True),
    )
    processor, storage, s3_client = _build_processor(
        indexing_service=indexing_service
    )
    key = "knowledge/raw/retry.txt"
    s3_client.add(key, b"hello")
    event = {"Records": [_record(key)]}

    with pytest.raises(IngestionBatchError) as first:
        processor.process_event(event)
    recovered = processor.process_event(event)

    assert isinstance(first.value.__cause__, AutomaticIndexingIncompleteError)
    assert recovered["records_processed"] == 1
    assert indexing_service.index_pending_document.call_count == 2
    manifest = storage.json_objects["knowledge/metadata/manifest.json"]
    assert len(manifest["documents"]) == 1


def test_multiple_records_continue_and_url_decode_object_keys():
    processor, _, s3_client = _build_processor()
    decoded_key = "knowledge/raw/design guide.md"
    other_key = "knowledge/raw/notes.txt"
    s3_client.add(decoded_key, b"# guide")
    s3_client.add(other_key, b"notes")

    result = processor.process_event(
        {
            "Records": [
                _record(decoded_key, size=7, version_id="version-1"),
                _record(other_key),
            ]
        }
    )

    assert result["records_received"] == 2
    assert result["records_processed"] == 2
    assert result["records_skipped"] == 0
    assert s3_client.get_calls[0] == {
        "Bucket": "knowledge-bucket",
        "Key": decoded_key,
        "VersionId": "version-1",
    }


def test_outside_prefix_folder_and_generated_output_are_skipped():
    processor, _, s3_client = _build_processor()

    result = processor.process_event(
        {
            "Records": [
                _record("other/location.txt"),
                _record("knowledge/raw/folder/"),
                _record("knowledge/processed/generated.txt"),
            ]
        }
    )

    assert result == {
        "records_received": 3,
        "records_processed": 0,
        "records_skipped": 3,
        "records_failed": 0,
        "indexable_documents_received": 0,
        "media_objects_stored": 0,
        "unsupported_binaries_stored": 0,
        "suspicious_objects_quarantined": 0,
        "media_indexing_attempts_blocked": 0,
        "mime_mismatch_count": 0,
    }
    assert s3_client.get_calls == []


def test_malformed_s3_events_and_records_are_safe_to_skip():
    processor, _, _ = _build_processor()

    assert processor.process_event({}) == _expected_summary(
        received=0, processed=0, skipped=0, failed=0
    )
    result = processor.process_event(
        {"Records": [None, {"eventSource": "not-s3"}]}
    )
    assert result == _expected_summary(
        received=2, processed=0, skipped=2, failed=0
    )


def test_duplicate_event_is_idempotently_skipped_without_new_writes():
    processor, storage, s3_client = _build_processor()
    key = "knowledge/raw/retry.json"
    s3_client.add(key, b'{"ok": true}')
    event = {"Records": [_record(key, size=12, version_id="version-7")]}

    first = processor.process_event(event)
    writes_after_first = storage.write_count
    second = processor.process_event(event)

    assert first["records_processed"] == 1
    assert second == _expected_summary(
        received=1,
        processed=0,
        skipped=1,
        failed=0,
        indexable=1,
    )
    assert storage.write_count == writes_after_first


def test_valid_pdf_event_creates_text_chunks_metadata_and_pending_embedding():
    processor, storage, s3_client = _build_processor()
    key = "knowledge/raw/architecture.pdf"
    content = make_text_pdf(["Architecture", "Deployment"])
    s3_client.add(key, content)

    result = processor.process_event(
        {"Records": [_record(key, size=len(content))]}
    )

    assert result == _expected_summary(
        received=1,
        processed=1,
        skipped=0,
        failed=0,
        indexable=1,
    )
    manifest = storage.json_objects["knowledge/metadata/manifest.json"]
    entry = next(iter(manifest["documents"].values()))
    assert entry["metadata"]["file_type"] == "pdf"
    assert entry["processed_key"] in storage.byte_objects
    assert entry["chunk_count"] > 0
    assert (
        storage.json_objects[entry["embedding_key"]]["status"]
        == "pending"
    )
    metadata = storage.json_objects[
        f"knowledge/metadata/{entry['document_id']}.json"
    ]["extraction"]
    assert metadata["page_count"] == 2
    assert metadata["pages_with_text"] == 2
    assert metadata["parser_library"] == "pypdf"


def test_mixed_pdf_batch_continues_and_reports_invalid_pdf_failure():
    processor, storage, s3_client = _build_processor()
    valid_pdf_key = "knowledge/raw/valid.pdf"
    invalid_pdf_key = "knowledge/raw/corrupt.pdf"
    text_key = "knowledge/raw/notes.txt"
    valid_pdf = make_text_pdf(["Valid PDF"])
    invalid_pdf = b"%PDF-1.7\ncorrupt"
    s3_client.add(valid_pdf_key, valid_pdf)
    s3_client.add(invalid_pdf_key, invalid_pdf)
    s3_client.add(text_key, b"valid text")

    with pytest.raises(IngestionBatchError) as raised:
        processor.process_event(
            {
                "Records": [
                    _record(valid_pdf_key, size=len(valid_pdf)),
                    _record(invalid_pdf_key, size=len(invalid_pdf)),
                    _record(text_key, size=10),
                ]
            }
        )

    assert raised.value.summary == _expected_summary(
        received=3,
        processed=2,
        skipped=0,
        failed=1,
        indexable=2,
    )
    manifest = storage.json_objects["knowledge/metadata/manifest.json"]
    assert len(manifest["documents"]) == 2
    assert isinstance(raised.value.__cause__, InvalidPdfError)


def test_duplicate_pdf_event_is_idempotently_skipped():
    processor, storage, s3_client = _build_processor()
    key = "knowledge/raw/retry.pdf"
    content = make_text_pdf(["Retry-safe PDF"])
    s3_client.add(key, content)
    event = {
        "Records": [
            _record(key, size=len(content), version_id="pdf-version-1")
        ]
    }

    first = processor.process_event(event)
    writes_after_first = storage.write_count
    second = processor.process_event(event)

    assert first["records_processed"] == 1
    assert second == _expected_summary(
        received=1,
        processed=0,
        skipped=1,
        failed=0,
        indexable=1,
    )
    assert storage.write_count == writes_after_first


def test_new_source_version_is_processed_independently():
    processor, storage, s3_client = _build_processor()
    key = "knowledge/raw/versioned.txt"
    s3_client.add(key, b"same content")

    first = processor.process_event(
        {"Records": [_record(key, size=12, version_id="version-1")]}
    )
    second = processor.process_event(
        {"Records": [_record(key, size=12, version_id="version-2")]}
    )

    assert first["records_processed"] == 1
    assert second["records_processed"] == 1
    manifest = storage.json_objects["knowledge/metadata/manifest.json"]
    assert len(manifest["documents"]) == 2


def test_existing_pipeline_source_metadata_reuses_its_document_id():
    processor, storage, s3_client = _build_processor()
    document_id = "0123456789abcdef0123456789abcdef"
    key = f"knowledge/raw/{document_id}/guide.txt"
    content = b"pipeline source"
    s3_client.add(
        key,
        content,
        metadata={
            "document-id": document_id,
            "checksum-sha256": hashlib.sha256(content).hexdigest(),
        },
    )

    first = processor.process_event(
        {"Records": [_record(key, size=len(content))]}
    )
    second = processor.process_event(
        {"Records": [_record(key, size=len(content))]}
    )

    assert first["records_processed"] == 1
    assert second["records_skipped"] == 1
    manifest = storage.json_objects["knowledge/metadata/manifest.json"]
    assert set(manifest["documents"]) == {document_id}


def test_one_record_failure_does_not_prevent_another_record_from_succeeding():
    processor, storage, s3_client = _build_processor()
    failed_key = "knowledge/raw/unavailable.txt"
    successful_key = "knowledge/raw/available.py"
    s3_client.fail_keys.add(failed_key)
    s3_client.add(successful_key, b"print('ok')")

    with pytest.raises(IngestionBatchError) as raised:
        processor.process_event(
            {
                "Records": [
                    _record(failed_key),
                    _record(successful_key, size=11),
                ]
            }
        )

    assert raised.value.summary == _expected_summary(
        received=2,
        processed=1,
        skipped=0,
        failed=1,
        indexable=1,
    )
    manifest = storage.json_objects["knowledge/metadata/manifest.json"]
    assert len(manifest["documents"]) == 1
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_oversized_text_is_classified_and_quarantined_without_indexing():
    processor, storage, s3_client = _build_processor()
    key = "knowledge/raw/huge.txt"
    content = b"x" * (10 * 1024 * 1024 + 1)
    s3_client.add(key, content, content_type="text/plain")

    result = processor.process_event(
        {
            "Records": [
                _record(
                    key,
                    size=10 * 1024 * 1024 + 1,
                )
            ]
        }
    )

    assert result["suspicious_objects_quarantined"] == 1
    assert result["media_indexing_attempts_blocked"] == 1
    assert len(s3_client.get_calls) == 1
    metadata = next(
        value
        for key, value in storage.json_objects.items()
        if key.startswith("knowledge/metadata/storage-only/")
    )
    assert metadata["quarantine_reason"] == "maximum_upload_size_exceeded"


def test_structured_logs_exclude_full_keys_and_document_contents(caplog):
    processor, _, s3_client = _build_processor()
    key = "knowledge/raw/private-filename.txt"
    content = b"sensitive document body"
    s3_client.add(key, content)
    caplog.set_level(logging.INFO, logger="test.document-ingestion")

    processor.process_event(
        {"Records": [_record(key, size=len(content))]}
    )

    events = [json.loads(record.message) for record in caplog.records]
    assert any(
        event.get("event") == "s3_knowledge_ingestion"
        and event.get("outcome") == "processed"
        for event in events
    )
    assert all(key not in record.message for record in caplog.records)
    assert all(
        content.decode("utf-8") not in record.message
        for record in caplog.records
    )


@pytest.mark.parametrize(
    ("filename", "content", "content_type", "media_type"),
    [
        ("photo.jpg", b"\xff\xd8\xff\xe0jpeg", "image/jpeg", "image"),
        (
            "diagram.png",
            b"\x89PNG\r\n\x1a\ncontent",
            "image/png",
            "image",
        ),
        (
            "clip.mp4",
            b"\x00\x00\x00\x18ftypisomvideo",
            "video/mp4",
            "video",
        ),
        ("audio.mp3", b"ID3audio", "audio/mpeg", "audio"),
    ],
)
def test_media_is_copied_to_scoped_storage_only_prefix_and_never_indexed(
    filename,
    content,
    content_type,
    media_type,
):
    indexing_service = Mock()
    processor, storage, s3_client = _build_processor(
        indexing_service=indexing_service
    )
    key = f"knowledge/raw/{filename}"
    s3_client.add(key, content, content_type=content_type)

    result = processor.process_event(
        {"Records": [_record(key, size=len(content))]}
    )

    assert result == _expected_summary(
        received=1,
        processed=1,
        skipped=0,
        failed=0,
        media=1,
        blocked=1,
    )
    indexing_service.index_pending_document.assert_not_called()
    assert len(s3_client.copy_calls) == 1
    copy_call = s3_client.copy_calls[0]
    assert str(copy_call["Key"]).startswith(
        f"knowledge/media/internal/dev/{media_type}/"
    )
    assert str(copy_call["Key"]).endswith(f"/{filename}")
    metadata_key = next(
        stored_key
        for stored_key in storage.json_objects
        if stored_key.startswith("knowledge/metadata/storage-only/")
    )
    metadata = storage.json_objects[metadata_key]
    assert metadata["original_filename"] == filename
    assert metadata["object_classification"] == "media_object"
    assert metadata["detected_mime_type"] == content_type
    assert metadata["declared_mime_type"] == content_type
    assert metadata["file_extension"] == filename.rsplit(".", 1)[1]
    assert metadata["media_type"] == media_type
    assert metadata["storage_only"] is True
    assert metadata["indexable"] is False
    assert metadata["quarantine_reason"] is None
    assert metadata["source_object_key"] == key
    assert metadata["source_s3_uri"].startswith("s3://knowledge-bucket/")
    assert metadata["checksum_sha256"] == hashlib.sha256(content).hexdigest()
    assert metadata["size_bytes"] == len(content)
    assert metadata["upload_timestamp"] == "2026-07-30T12:00:00Z"
    assert "knowledge/metadata/manifest.json" not in storage.json_objects
    assert not any(
        stored_key.startswith(
            (
                "knowledge/chunks/",
                "knowledge/embeddings/",
                "knowledge/processed/",
            )
        )
        for stored_key in (
            set(storage.json_objects) | set(storage.byte_objects)
        )
    )


def test_required_media_extension_sets_are_complete():
    assert IMAGE_EXTENSIONS == {
        "jpg",
        "jpeg",
        "png",
        "gif",
        "webp",
        "bmp",
        "tiff",
        "svg",
        "heic",
    }
    assert VIDEO_EXTENSIONS == {
        "mp4",
        "mov",
        "avi",
        "mkv",
        "webm",
        "mpeg",
        "mpg",
        "m4v",
    }
    assert AUDIO_EXTENSIONS == {
        "mp3",
        "wav",
        "m4a",
        "aac",
        "flac",
        "ogg",
    }
    assert MEDIA_EXTENSIONS == (
        IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
    )


@pytest.mark.parametrize(
    ("filename", "content", "content_type", "reason", "mime_mismatch"),
    [
        (
            "renamed.txt",
            b"\xff\xd8\xff\xe0jpeg",
            "text/plain",
            "extension_signature_mismatch",
            0,
        ),
        (
            "renamed.jpg",
            b"plain UTF-8 text",
            "image/jpeg",
            "extension_signature_mismatch",
            0,
        ),
        (
            "wrong-mime.png",
            b"\x89PNG\r\n\x1a\ncontent",
            "image/jpeg",
            "declared_mime_type_mismatch",
            1,
        ),
    ],
)
def test_extension_signature_and_mime_conflicts_are_quarantined(
    filename,
    content,
    content_type,
    reason,
    mime_mismatch,
    caplog,
):
    indexing_service = Mock()
    processor, storage, s3_client = _build_processor(
        indexing_service=indexing_service
    )
    key = f"knowledge/raw/{filename}"
    s3_client.add(key, content, content_type=content_type)
    caplog.set_level(logging.WARNING, logger="test.document-ingestion")

    result = processor.process_event(
        {"Records": [_record(key, size=len(content))]}
    )

    assert result == _expected_summary(
        received=1,
        processed=1,
        skipped=0,
        failed=0,
        suspicious=1,
        blocked=1,
        mime_mismatch=mime_mismatch,
    )
    indexing_service.index_pending_document.assert_not_called()
    assert str(s3_client.copy_calls[0]["Key"]).startswith(
        "knowledge/quarantine/internal/dev/"
    )
    metadata = next(
        value
        for stored_key, value in storage.json_objects.items()
        if stored_key.startswith("knowledge/metadata/storage-only/")
    )
    assert metadata["object_classification"] == "rejected_or_suspicious"
    assert metadata["quarantine_reason"] == reason
    warning = json.loads(caplog.records[-1].message)
    assert warning["reason"] == "suspicious_object_quarantined"
    assert warning["quarantine_reason"] == reason
    assert key not in caplog.text


def test_unsupported_binary_is_storage_only_and_never_reaches_pipeline():
    processor, storage, s3_client = _build_processor()
    key = "knowledge/raw/archive.bin"
    content = b"\x00\x01\x02\x03unsupported"
    s3_client.add(key, content, content_type="application/octet-stream")

    with patch.object(
        processor._pipeline,
        "ingest_existing_raw",
    ) as ingest_existing_raw:
        result = processor.process_event(
            {"Records": [_record(key, size=len(content))]}
        )

    assert result == _expected_summary(
        received=1,
        processed=1,
        skipped=0,
        failed=0,
        unsupported=1,
        blocked=1,
    )
    ingest_existing_raw.assert_not_called()
    assert str(s3_client.copy_calls[0]["Key"]).startswith(
        "knowledge/media/internal/dev/other/"
    )
    metadata = next(
        value
        for stored_key, value in storage.json_objects.items()
        if stored_key.startswith("knowledge/metadata/storage-only/")
    )
    assert metadata["object_classification"] == "unsupported_binary"
    assert metadata["storage_only"] is True
    assert metadata["indexable"] is False


HANDLER_PATH = (
    Path(__file__).parents[2]
    / "lambda"
    / "document_ingestion"
    / "index.py"
)
HANDLER_SPEC = importlib.util.spec_from_file_location(
    "document_ingestion_handler",
    HANDLER_PATH,
)
document_ingestion_handler = importlib.util.module_from_spec(HANDLER_SPEC)
HANDLER_SPEC.loader.exec_module(document_ingestion_handler)


def test_lambda_handler_delegates_without_constructing_bedrock_client():
    processor = Mock()
    processor.process_event.return_value = {
        "records_received": 1,
        "records_processed": 1,
        "records_skipped": 0,
        "records_failed": 0,
    }
    event = {"Records": [_record("knowledge/raw/guide.txt")]}

    with (
        patch.object(document_ingestion_handler, "_processor", processor),
        patch.object(
            document_ingestion_handler.boto3,
            "client",
        ) as boto3_client,
    ):
        result = document_ingestion_handler.handler(event, Mock())

    assert result["records_processed"] == 1
    processor.process_event.assert_called_once_with(event)
    boto3_client.assert_not_called()


def test_lambda_factory_constructs_only_an_s3_client():
    s3_client = Mock()
    environment = {
        "KNOWLEDGE_BUCKET_NAME": "knowledge-bucket",
        "KNOWLEDGE_RAW_PREFIX": "knowledge/raw/",
        "KNOWLEDGE_SUPPORTED_DOCUMENT_TYPES": "txt,md,json,html,py",
        "KNOWLEDGE_CHUNK_SIZE": "1000",
        "KNOWLEDGE_CHUNK_OVERLAP": "100",
        "KNOWLEDGE_MAXIMUM_UPLOAD_SIZE": "10485760",
        "CLIENT_ID": "internal",
        "DEPLOYMENT_ENVIRONMENT": "dev",
    }

    with (
        patch.dict(
            document_ingestion_handler.os.environ,
            environment,
            clear=True,
        ),
        patch.object(
            document_ingestion_handler.boto3,
            "client",
            return_value=s3_client,
        ) as boto3_client,
    ):
        processor = document_ingestion_handler._build_processor()

    assert isinstance(processor, S3DocumentIngestionProcessor)
    boto3_client.assert_called_once_with("s3")


def test_lambda_factory_composes_explicit_offline_indexing_without_network():
    s3_client = Mock()
    environment = {
        "KNOWLEDGE_BUCKET_NAME": "knowledge-bucket",
        "KNOWLEDGE_RAW_PREFIX": "knowledge/raw/",
        "KNOWLEDGE_SUPPORTED_DOCUMENT_TYPES": "txt,md,json,html,py",
        "KNOWLEDGE_CHUNK_SIZE": "1000",
        "KNOWLEDGE_CHUNK_OVERLAP": "100",
        "KNOWLEDGE_MAXIMUM_UPLOAD_SIZE": "10485760",
        "KNOWLEDGE_AUTOMATIC_INDEXING_ENABLED": "true",
        "KNOWLEDGE_EMBEDDING_PROVIDER": "fake",
        "KNOWLEDGE_VECTOR_STORE_PROVIDER": "memory",
        "KNOWLEDGE_EMBEDDING_MODEL_ID": "offline-index-v1",
        "CLIENT_ID": "internal",
        "DEPLOYMENT_ENVIRONMENT": "dev",
    }

    with (
        patch.dict(
            document_ingestion_handler.os.environ,
            environment,
            clear=True,
        ),
        patch.object(
            document_ingestion_handler.boto3,
            "client",
            return_value=s3_client,
        ) as boto3_client,
    ):
        processor = document_ingestion_handler._build_processor()

    assert processor._indexing_service is not None
    boto3_client.assert_called_once_with("s3")
