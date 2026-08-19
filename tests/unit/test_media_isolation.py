from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping
from unittest.mock import Mock

import pytest

from knowledge.config import KnowledgeConfig
from knowledge.embedding_workflow import EmbeddingWorkflow
from knowledge.ingestion import KnowledgeIngestionPipeline
from knowledge.media_classification import (
    NonIndexableObjectError,
    ObjectClassification,
    classify_uploaded_object,
)
from knowledge.models import (
    DocumentMetadata,
    EmbeddingRecord,
    KnowledgeManifestEntry,
)
from knowledge.qdrant_vector_store import QdrantVectorStore
from knowledge.retrieval import RetrievalEntry
from knowledge.vector_indexing import VectorIndexingWorkflow
from knowledge.vector_store import InMemoryVectorStore


FIXED_TIME = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)

MEDIA_SIGNATURE_CASES = [
    ("jpg", b"\xff\xd8\xff\xe0jpeg", "image"),
    ("jpeg", b"\xff\xd8\xff\xe0jpeg", "image"),
    ("png", b"\x89PNG\r\n\x1a\ncontent", "image"),
    ("gif", b"GIF89acontent", "image"),
    ("webp", b"RIFF\x10\x00\x00\x00WEBPcontent", "image"),
    ("bmp", b"BMbitmap", "image"),
    ("tiff", b"II*\x00tiff", "image"),
    ("svg", b'<svg xmlns="http://www.w3.org/2000/svg"></svg>', "image"),
    ("heic", b"\x00\x00\x00\x18ftypheiccontent", "image"),
    ("mp4", b"\x00\x00\x00\x18ftypisomvideo", "video"),
    ("mov", b"\x00\x00\x00\x18ftypqt  video", "video"),
    ("avi", b"RIFF\x10\x00\x00\x00AVI content", "video"),
    ("mkv", b"\x1aE\xdf\xa3matroska", "video"),
    ("webm", b"\x1aE\xdf\xa3webm", "video"),
    ("mpeg", b"\x00\x00\x01\xbavideo", "video"),
    ("mpg", b"\x00\x00\x01\xb3video", "video"),
    ("m4v", b"\x00\x00\x00\x18ftypisomvideo", "video"),
    ("mp3", b"ID3audio", "audio"),
    ("wav", b"RIFF\x10\x00\x00\x00WAVEaudio", "audio"),
    ("m4a", b"\x00\x00\x00\x18ftypM4A audio", "audio"),
    ("aac", b"ADIFaudio", "audio"),
    ("flac", b"fLaCaudio", "audio"),
    ("ogg", b"OggSaudio", "audio"),
]


class MemoryStorage:
    def __init__(self) -> None:
        self.byte_objects: dict[str, bytes] = {}
        self.json_objects: dict[str, dict[str, Any]] = {}

    def put_bytes(
        self,
        key: str,
        content: bytes,
        *,
        content_type: str,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        del content_type, metadata
        self.byte_objects[key] = content

    def put_json(self, key: str, payload: Mapping[str, Any]) -> None:
        self.json_objects[key] = deepcopy(dict(payload))

    def get_json(self, key: str) -> dict[str, Any] | None:
        value = self.json_objects.get(key)
        return deepcopy(value) if value is not None else None


@pytest.mark.parametrize(
    ("extension", "content", "expected_media_type"),
    MEDIA_SIGNATURE_CASES,
)
def test_every_approved_media_extension_is_storage_only(
    extension,
    content,
    expected_media_type,
):
    inspection = classify_uploaded_object(
        filename=f"object.{extension}",
        declared_mime_type="application/octet-stream",
        content=content,
    )

    assert inspection.object_classification is ObjectClassification.MEDIA_OBJECT
    assert inspection.media_type is not None
    assert inspection.media_type.value == expected_media_type
    assert inspection.storage_only is True
    assert inspection.indexable is False
    assert inspection.quarantine_reason is None


def test_pipeline_guards_extraction_and_chunking_for_media():
    storage = MemoryStorage()
    extractors = Mock()
    chunker = Mock()
    pipeline = KnowledgeIngestionPipeline(
        storage,
        KnowledgeConfig(supported_document_types=frozenset({"jpg"})),
        extractors=extractors,
        chunker=chunker,
        clock=lambda: FIXED_TIME,
        document_id_factory=lambda: "media-document",
    )

    with pytest.raises(NonIndexableObjectError, match="text_extraction"):
        pipeline.ingest(
            filename="photo.jpg",
            content=b"\xff\xd8\xff\xe0jpeg",
            source="local-upload",
            declared_mime_type="image/jpeg",
        )

    extractors.extract_with_metadata.assert_not_called()
    chunker.chunk.assert_not_called()
    assert not any(key.startswith("knowledge/chunks/") for key in storage.json_objects)
    assert not any(
        key.startswith("knowledge/embeddings/") for key in storage.json_objects
    )


def test_embedding_guard_blocks_media_before_provider_or_storage_access():
    provider = Mock(provider_name="fake")
    storage = Mock()
    workflow = EmbeddingWorkflow(
        storage=storage,
        provider=provider,
        model_id="fake-v1",
        batch_size=1,
    )

    with pytest.raises(NonIndexableObjectError, match="embedding_generation"):
        workflow.embed_document(
            document_id="media-document",
            chunks=(),
            source_object_key="knowledge/media/client/dev/image/photo.jpg",
            object_classification="media_object",
        )

    provider.embed.assert_not_called()
    storage.get_json.assert_not_called()
    storage.put_json.assert_not_called()


def test_vector_indexing_guard_blocks_media_before_chunks_or_upsert():
    storage = Mock()
    embedding_workflow = Mock()
    vector_store = Mock()
    manifest = Mock()
    metadata = DocumentMetadata(
        filename="photo.jpg",
        file_type="jpg",
        upload_timestamp="2026-08-05T12:00:00Z",
        checksum="abc",
        source="s3://bucket/knowledge/raw/photo.jpg",
        document_size=10,
        object_classification="media_object",
        detected_mime_type="image/jpeg",
        declared_mime_type="image/jpeg",
        file_extension="jpg",
        media_type="image",
        storage_only=True,
        indexable=False,
    )
    entry = KnowledgeManifestEntry(
        document_id="media-document",
        metadata=metadata,
        chunk_count=0,
        embedding_status="pending",
        ingestion_timestamp=metadata.upload_timestamp,
        raw_key="knowledge/media/client/dev/image/photo.jpg",
        processed_key=None,
        chunks_key="knowledge/chunks/media-document.json",
        embedding_key="knowledge/embeddings/media-document.json",
    )
    workflow = VectorIndexingWorkflow(
        storage=storage,
        embedding_workflow=embedding_workflow,
        vector_store=vector_store,
        manifest=manifest,
    )

    with pytest.raises(NonIndexableObjectError, match="indexing_chunk_load"):
        workflow.index_pending_document(
            entry,
            client_id="client",
            environment="dev",
        )

    storage.get_json.assert_not_called()
    embedding_workflow.embed_document.assert_not_called()
    vector_store.upsert.assert_not_called()


def test_classifier_rejects_executable_signature_even_with_text_extension():
    inspection = classify_uploaded_object(
        filename="payload.txt",
        declared_mime_type="text/plain",
        content=b"MZ\x90\x00binary",
        supported_document_types={"txt"},
    )

    assert inspection.object_classification is (
        ObjectClassification.REJECTED_OR_SUSPICIOUS
    )
    assert inspection.quarantine_reason == "dangerous_executable_signature"
    assert inspection.storage_only is True
    assert inspection.indexable is False


@pytest.mark.parametrize("store_name", ["memory", "qdrant"])
def test_vector_store_adapters_reject_media_before_persistence(store_name):
    client = Mock()
    store = (
        InMemoryVectorStore()
        if store_name == "memory"
        else QdrantVectorStore(client=client, models_module=Mock())
    )
    entry = RetrievalEntry(
        embedding_record=EmbeddingRecord(
            schema_version=EmbeddingRecord.CURRENT_SCHEMA_VERSION,
            document_id="media-document",
            chunk_id="media-document:000000",
            chunk_text_checksum="a" * 64,
            embedding_model_id="should-not-exist",
            embedding_dimensions=2,
            embedding_vector=(1.0, 0.0),
            creation_timestamp="2026-08-05T12:00:00Z",
            source_object_key="knowledge/media/client/dev/image/photo.jpg",
        ),
        source="storage-only media",
        text="media must never have retrieval text",
        metadata={
            "client_id": "client",
            "environment": "dev",
            "object_classification": "media_object",
            "indexable": False,
            "storage_only": True,
        },
    )

    with pytest.raises(NonIndexableObjectError, match="vector_store_upsert"):
        store.upsert([entry], client_id="client", environment="dev")

    client.collection_exists.assert_not_called()
    client.upsert.assert_not_called()
