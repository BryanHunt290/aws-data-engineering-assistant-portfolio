from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import logging
from typing import Any, Mapping
from unittest.mock import Mock

import pytest

from knowledge.chunking import Chunker, TextChunker
from knowledge.config import (
    DEFAULT_SUPPORTED_DOCUMENT_TYPES,
    KnowledgeConfig,
)
from knowledge.embeddings import EmbeddingProvider, EmbeddingStatus
from knowledge.ingestion import KnowledgeIngestionPipeline
from knowledge.manifest import KnowledgeManifestRepository
from knowledge.models import DocumentMetadata, KnowledgeManifestEntry
from knowledge.storage import (
    FileSystemKnowledgeStorage,
    KnowledgeKeys,
    S3KnowledgeStorage,
)
from tests.unit.pdf_fixtures import make_text_pdf


class MemoryKnowledgeStorage:
    def __init__(self) -> None:
        self.byte_objects: dict[str, dict[str, Any]] = {}
        self.json_objects: dict[str, dict[str, Any]] = {}

    def put_bytes(
        self,
        key: str,
        content: bytes,
        *,
        content_type: str,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        self.byte_objects[key] = {
            "content": content,
            "content_type": content_type,
            "metadata": dict(metadata or {}),
        }

    def put_json(self, key: str, payload: Mapping[str, Any]) -> None:
        self.json_objects[key] = deepcopy(dict(payload))

    def get_json(self, key: str) -> dict[str, Any] | None:
        payload = self.json_objects.get(key)
        return deepcopy(payload) if payload is not None else None


FIXED_TIME = datetime(2026, 7, 27, 12, 30, tzinfo=timezone.utc)


def _pipeline(
    storage: MemoryKnowledgeStorage,
    *,
    config: KnowledgeConfig | None = None,
    event_logger: logging.Logger | None = None,
) -> KnowledgeIngestionPipeline:
    return KnowledgeIngestionPipeline(
        storage,
        config,
        clock=lambda: FIXED_TIME,
        document_id_factory=lambda: "document-123",
        event_logger=event_logger,
    )


def _manifest_entry(document_id: str) -> KnowledgeManifestEntry:
    metadata = DocumentMetadata(
        filename=f"{document_id}.txt",
        file_type="txt",
        upload_timestamp="2026-07-27T12:30:00Z",
        checksum="abc123",
        source="unit-test",
        document_size=3,
    )
    return KnowledgeManifestEntry(
        document_id=document_id,
        metadata=metadata,
        chunk_count=1,
        embedding_status=EmbeddingStatus.PENDING.value,
        ingestion_timestamp=metadata.upload_timestamp,
        raw_key=KnowledgeKeys.raw(document_id, metadata.filename),
        processed_key=KnowledgeKeys.processed(document_id),
        chunks_key=KnowledgeKeys.chunks(document_id),
        embedding_key=KnowledgeKeys.embeddings(document_id),
    )


def test_knowledge_config_has_required_defaults_and_document_types():
    config = KnowledgeConfig()

    assert config.chunk_size == 1_000
    assert config.overlap == 100
    assert config.maximum_upload_size == 10 * 1024 * 1024
    assert {
        "pdf",
        "md",
        "txt",
        "docx",
        "html",
        "json",
        "py",
    }.issubset(DEFAULT_SUPPORTED_DOCUMENT_TYPES)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"chunk_size": 0}, "chunk_size"),
        ({"chunk_size": 10, "overlap": 10}, "overlap"),
        ({"overlap": -1}, "overlap"),
        ({"maximum_upload_size": 0}, "maximum_upload_size"),
        ({"supported_document_types": frozenset()}, "supported_document_types"),
    ],
)
def test_knowledge_config_rejects_invalid_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        KnowledgeConfig(**kwargs)


def test_text_chunker_preserves_document_references_and_overlap():
    chunker: Chunker = TextChunker(chunk_size=4, overlap=1)

    chunks = chunker.chunk("document-123", "abcdefghij")

    assert [chunk.text for chunk in chunks] == ["abcd", "defg", "ghij"]
    assert [chunk.document_id for chunk in chunks] == [
        "document-123",
        "document-123",
        "document-123",
    ]
    assert [
        (chunk.start_character, chunk.end_character) for chunk in chunks
    ] == [(0, 4), (3, 7), (6, 10)]
    assert chunks[0].chunk_id == "document-123:000000"


def test_text_chunker_returns_no_chunks_for_empty_text():
    assert TextChunker(100, 10).chunk("document-123", "") == []


def test_embedding_interface_accepts_future_provider_without_bedrock_calls():
    class FakeEmbeddingProvider:
        provider_name = "fake"

        def embed(self, texts):
            return [[float(len(text))] for text in texts]

    provider = FakeEmbeddingProvider()

    assert isinstance(provider, EmbeddingProvider)
    assert provider.embed(["one", "three"]) == [[3.0], [5.0]]


def test_filesystem_storage_round_trips_json_and_rejects_escape(tmp_path):
    storage = FileSystemKnowledgeStorage(tmp_path / "artifacts")

    storage.put_json("knowledge/metadata/test.json", {"status": "pending"})

    assert storage.get_json("knowledge/metadata/test.json") == {
        "status": "pending"
    }
    with pytest.raises(ValueError, match="escape"):
        storage.put_json("../outside.json", {"secret": False})


def test_ingestion_preserves_original_and_writes_full_knowledge_hierarchy():
    storage = MemoryKnowledgeStorage()
    pipeline = _pipeline(
        storage,
        config=KnowledgeConfig(chunk_size=8, overlap=2),
    )
    content = b"knowledge layer text"

    entry = pipeline.ingest(
        filename="guide.txt",
        content=content,
        source="manual-upload",
    )

    raw_key = "knowledge/raw/document-123/guide.txt"
    assert storage.byte_objects[raw_key]["content"] == content
    assert storage.byte_objects[raw_key]["metadata"] == {
        "checksum-sha256": (
            "83106ff0c7ed27cf3adb55786924227e"
            "05c54a8bfac7763f4fe35be8050bcef7"
        ),
        "document-id": "document-123",
    }
    assert storage.byte_objects[
        "knowledge/processed/document-123.txt"
    ]["content"] == content

    metadata_payload = storage.json_objects[
        "knowledge/metadata/document-123.json"
    ]
    assert metadata_payload["metadata"] == {
        "filename": "guide.txt",
        "file_type": "txt",
        "upload_timestamp": "2026-07-27T12:30:00Z",
        "checksum": (
            "83106ff0c7ed27cf3adb55786924227e"
            "05c54a8bfac7763f4fe35be8050bcef7"
        ),
            "source": "manual-upload",
            "document_size": len(content),
            "object_classification": "indexable_text_document",
            "detected_mime_type": "text/plain",
            "declared_mime_type": "text/plain",
            "file_extension": "txt",
            "media_type": None,
            "storage_only": False,
            "indexable": True,
            "quarantine_reason": None,
            "source_s3_uri": None,
            "checksum_sha256": (
                "83106ff0c7ed27cf3adb55786924227e"
                "05c54a8bfac7763f4fe35be8050bcef7"
            ),
            "size_bytes": len(content),
        }

    chunks_payload = storage.json_objects[
        "knowledge/chunks/document-123.json"
    ]
    assert len(chunks_payload["chunks"]) == 3
    assert all(
        chunk["document_id"] == "document-123"
        for chunk in chunks_payload["chunks"]
    )

    embedding_payload = storage.json_objects[
        "knowledge/embeddings/document-123.json"
    ]
    assert embedding_payload["document_id"] == "document-123"
    assert embedding_payload["provider"] is None
    assert embedding_payload["status"] == "pending"
    assert embedding_payload["index_status"] == "pending"
    assert embedding_payload["vectors"] == []
    assert len(embedding_payload["chunks"]) == 3
    assert all(
        state["status"] == "pending"
        and state["attempt_count"] == 0
        and len(state["checksum"]) == 64
        for state in embedding_payload["chunks"]
    )

    manifest = storage.json_objects[KnowledgeKeys.MANIFEST]
    assert manifest["schema_version"] == 1
    assert manifest["documents"]["document-123"] == entry.to_dict()
    assert entry.chunk_count == 3
    assert entry.embedding_status == "pending"


def test_direct_pdf_ingestion_uses_existing_pipeline_contracts():
    storage = MemoryKnowledgeStorage()
    content = make_text_pdf(["Architecture guide", "Operations guide"])

    entry = _pipeline(storage).ingest(
        filename="design.pdf",
        content=content,
        source="project-docs",
    )

    assert (
        storage.byte_objects[
            "knowledge/raw/document-123/design.pdf"
        ]["content"]
        == content
    )
    assert entry.processed_key == "knowledge/processed/document-123.txt"
    assert entry.chunk_count > 0
    assert storage.byte_objects[entry.processed_key]["content"].startswith(
        b"Architecture guide"
    )
    metadata = storage.json_objects[
        "knowledge/metadata/document-123.json"
    ]
    assert metadata["metadata"]["checksum"] == hashlib.sha256(
        content
    ).hexdigest()
    assert metadata["extraction"]["page_count"] == 2
    assert metadata["extraction"]["pages_with_text"] == 2
    assert "Architecture guide" not in json.dumps(metadata)
    assert storage.json_objects[entry.embedding_key]["status"] == "pending"


def test_direct_pdf_respects_existing_maximum_upload_size():
    storage = MemoryKnowledgeStorage()
    content = make_text_pdf(["Oversized PDF"])
    config = KnowledgeConfig(maximum_upload_size=len(content) - 1)

    with pytest.raises(ValueError, match="maximum_upload_size"):
        _pipeline(storage, config=config).ingest(
            filename="oversized.pdf",
            content=content,
            source="unit-test",
        )

    assert not storage.byte_objects
    assert not storage.json_objects


def test_ingestion_processes_an_existing_raw_object_without_copying_it():
    storage = MemoryKnowledgeStorage()

    entry = _pipeline(storage).ingest_existing_raw(
        document_id="event-document",
        filename="guide.txt",
        content=b"already uploaded",
        source="s3://knowledge-bucket/knowledge/raw/guide.txt",
        raw_key="knowledge/raw/guide.txt",
    )

    assert entry.document_id == "event-document"
    assert entry.raw_key == "knowledge/raw/guide.txt"
    assert not any(
        key.startswith("knowledge/raw/") for key in storage.byte_objects
    )
    assert (
        storage.json_objects["knowledge/metadata/event-document.json"][
            "raw_key"
        ]
        == "knowledge/raw/guide.txt"
    )


@pytest.mark.parametrize(
    ("filename", "content", "source", "error"),
    [
        ("unsupported.exe", b"x", "test", "Unsupported document type"),
        ("../guide.txt", b"x", "test", "path components"),
        ("guide.txt", b"x", "", "source cannot be empty"),
        ("guide.txt", "not-bytes", "test", "content must be bytes"),
    ],
)
def test_ingestion_rejects_invalid_document_inputs(
    filename,
    content,
    source,
    error,
):
    with pytest.raises((TypeError, ValueError), match=error):
        _pipeline(MemoryKnowledgeStorage()).ingest(
            filename=filename,
            content=content,
            source=source,
        )


def test_ingestion_enforces_maximum_upload_size():
    config = KnowledgeConfig(maximum_upload_size=3)

    with pytest.raises(ValueError, match="maximum_upload_size"):
        _pipeline(
            MemoryKnowledgeStorage(),
            config=config,
        ).ingest(
            filename="guide.txt",
            content=b"four",
            source="test",
        )


def test_manifest_preserves_every_document_entry():
    storage = MemoryKnowledgeStorage()
    repository = KnowledgeManifestRepository(storage)

    repository.upsert(_manifest_entry("first"))
    repository.upsert(_manifest_entry("second"))

    manifest = storage.json_objects[KnowledgeKeys.MANIFEST]
    assert set(manifest["documents"]) == {"first", "second"}
    assert repository.get("first")["document_id"] == "first"
    assert repository.get("missing") is None


def test_manifest_embedding_status_can_be_updated_independently():
    storage = MemoryKnowledgeStorage()
    repository = KnowledgeManifestRepository(storage)
    repository.upsert(_manifest_entry("first"))

    repository.update_embedding_status("first", "complete")

    assert repository.get("first")["embedding_status"] == "complete"


def test_ingestion_emits_structured_success_logs_for_every_step(caplog):
    event_logger = logging.getLogger("test.knowledge.success")
    caplog.set_level(logging.INFO, logger=event_logger.name)

    _pipeline(
        MemoryKnowledgeStorage(),
        event_logger=event_logger,
    ).ingest(
        filename="guide.txt",
        content=b"hello",
        source="test",
    )

    events = [json.loads(record.message) for record in caplog.records]
    assert {event["step"] for event in events} == {
        "metadata_extraction",
        "raw_upload",
        "metadata_upload",
        "text_extraction",
        "processed_upload",
        "chunking",
        "chunks_upload",
        "embedding_status_upload",
        "manifest_update",
        "ingestion",
    }
    assert all(event["document_id"] == "document-123" for event in events)
    assert all(event["success"] is True for event in events)
    assert all(event["elapsed_ms"] >= 0 for event in events)


def test_ingestion_emits_structured_failure_logs(caplog):
    event_logger = logging.getLogger("test.knowledge.failure")
    caplog.set_level(logging.INFO, logger=event_logger.name)

    with pytest.raises(ValueError, match="Unsupported document type"):
        _pipeline(
            MemoryKnowledgeStorage(),
            event_logger=event_logger,
        ).ingest(
            filename="malware.exe",
            content=b"x",
            source="test",
        )

    events = [json.loads(record.message) for record in caplog.records]
    assert [event["step"] for event in events] == [
        "metadata_extraction",
        "ingestion",
    ]
    assert all(event["success"] is False for event in events)
    assert all(event["error_type"] == "ValueError" for event in events)


def test_s3_storage_writes_bytes_and_json_with_expected_parameters():
    s3_client = Mock()
    storage = S3KnowledgeStorage("knowledge-bucket", s3_client)

    storage.put_bytes(
        "knowledge/raw/id/file.txt",
        b"original",
        content_type="text/plain",
        metadata={"document-id": "id"},
    )
    storage.put_json("knowledge/metadata/id.json", {"document_id": "id"})

    first_call, second_call = s3_client.put_object.call_args_list
    assert first_call.kwargs == {
        "Bucket": "knowledge-bucket",
        "Key": "knowledge/raw/id/file.txt",
        "Body": b"original",
        "ContentType": "text/plain",
        "Metadata": {"document-id": "id"},
    }
    assert second_call.kwargs["Bucket"] == "knowledge-bucket"
    assert second_call.kwargs["Key"] == "knowledge/metadata/id.json"
    assert second_call.kwargs["ContentType"] == "application/json"
    assert json.loads(second_call.kwargs["Body"]) == {"document_id": "id"}


def test_s3_storage_reads_json_and_handles_missing_keys():
    s3_client = Mock()
    s3_client.get_object.return_value = {
        "Body": BytesIO(b'{"documents": {}}')
    }
    storage = S3KnowledgeStorage("knowledge-bucket", s3_client)

    assert storage.get_json(KnowledgeKeys.MANIFEST) == {"documents": {}}

    class MissingKeyError(Exception):
        response = {"Error": {"Code": "NoSuchKey"}}

    s3_client.get_object.side_effect = MissingKeyError()
    assert storage.get_json("missing.json") is None


def test_s3_storage_uses_prefix_list_to_distinguish_missing_access_denied():
    s3_client = Mock()

    class AccessDeniedError(Exception):
        response = {"Error": {"Code": "AccessDenied"}}

    s3_client.get_object.side_effect = AccessDeniedError()
    s3_client.list_objects_v2.return_value = {"Contents": []}
    storage = S3KnowledgeStorage("knowledge-bucket", s3_client)

    assert storage.get_json("knowledge/metadata/manifest.json") is None
    s3_client.list_objects_v2.assert_called_once_with(
        Bucket="knowledge-bucket",
        Prefix="knowledge/metadata/manifest.json",
        MaxKeys=1,
    )


def test_s3_storage_does_not_hide_access_denied_for_an_existing_object():
    s3_client = Mock()

    class AccessDeniedError(Exception):
        response = {"Error": {"Code": "AccessDenied"}}

    s3_client.get_object.side_effect = AccessDeniedError()
    s3_client.list_objects_v2.return_value = {
        "Contents": [{"Key": "knowledge/metadata/manifest.json"}]
    }
    storage = S3KnowledgeStorage("knowledge-bucket", s3_client)

    with pytest.raises(AccessDeniedError):
        storage.get_json("knowledge/metadata/manifest.json")
