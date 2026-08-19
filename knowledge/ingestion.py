"""S3-oriented knowledge ingestion orchestration."""

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import hashlib
import logging
import mimetypes
from pathlib import PurePath
from time import perf_counter
import uuid

from knowledge.chunking import Chunker, TextChunker
from knowledge.config import KnowledgeConfig
from knowledge.embeddings import EmbeddingStatus
from knowledge.extraction import ExtractorRegistry
from knowledge.logging import IngestionLogger
from knowledge.manifest import KnowledgeManifestRepository
from knowledge.media_classification import (
    ObjectInspection,
    classify_uploaded_object,
    require_indexable_object,
)
from knowledge.models import (
    DocumentMetadata,
    KnowledgeChunk,
    KnowledgeManifestEntry,
    chunk_text_checksum,
)
from knowledge.storage import KnowledgeKeys, KnowledgeStorage


logger = logging.getLogger(__name__)


class KnowledgeIngestionPipeline:
    """Preserve, describe, chunk, and index one uploaded document."""

    def __init__(
        self,
        storage: KnowledgeStorage,
        config: KnowledgeConfig | None = None,
        *,
        chunker: Chunker | None = None,
        extractors: ExtractorRegistry | None = None,
        manifest: KnowledgeManifestRepository | None = None,
        clock: Callable[[], datetime] | None = None,
        document_id_factory: Callable[[], str] | None = None,
        event_logger: logging.Logger | None = None,
    ) -> None:
        self._storage = storage
        self._config = config or KnowledgeConfig()
        self._chunker = chunker or TextChunker(
            self._config.chunk_size,
            self._config.overlap,
        )
        self._extractors = extractors or ExtractorRegistry()
        self._manifest = manifest or KnowledgeManifestRepository(storage)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._document_id_factory = (
            document_id_factory or (lambda: uuid.uuid4().hex)
        )
        self._log = IngestionLogger(event_logger or logger)

    def ingest(
        self,
        *,
        filename: str,
        content: bytes,
        source: str,
        declared_mime_type: str | None = None,
    ) -> KnowledgeManifestEntry:
        """Ingest one document without invoking an embedding model."""

        inspection = classify_uploaded_object(
            filename=filename,
            declared_mime_type=(
                declared_mime_type or mimetypes.guess_type(filename)[0]
            ),
            content=content,
            supported_document_types=self._config.supported_document_types,
        )
        return self._ingest(
            document_id=self._document_id_factory(),
            filename=filename,
            content=content,
            source=source,
            existing_raw_key=None,
            inspection=inspection,
        )

    def ingest_existing_raw(
        self,
        *,
        document_id: str,
        filename: str,
        content: bytes,
        source: str,
        raw_key: str,
        inspection: ObjectInspection | None = None,
        declared_mime_type: str | None = None,
    ) -> KnowledgeManifestEntry:
        """Process an object that is already preserved under the raw prefix."""

        if not document_id.strip():
            raise ValueError("document_id cannot be empty")
        if not raw_key.strip():
            raise ValueError("raw_key cannot be empty")
        resolved_inspection = inspection or classify_uploaded_object(
            filename=filename,
            declared_mime_type=declared_mime_type,
            content=content,
            supported_document_types=self._config.supported_document_types,
        )
        return self._ingest(
            document_id=document_id,
            filename=filename,
            content=content,
            source=source,
            existing_raw_key=raw_key,
            inspection=resolved_inspection,
        )

    def _ingest(
        self,
        *,
        document_id: str,
        filename: str,
        content: bytes,
        source: str,
        existing_raw_key: str | None,
        inspection: ObjectInspection,
    ) -> KnowledgeManifestEntry:
        ingestion_started = perf_counter()
        try:
            metadata = self._log.run_step(
                document_id,
                "metadata_extraction",
                lambda: self._extract_metadata(
                    filename=filename,
                    content=content,
                    source=source,
                    inspection=inspection,
                ),
            )
            raw_key = existing_raw_key or KnowledgeKeys.raw(
                document_id,
                metadata.filename,
            )
            if existing_raw_key is None:
                self._log.run_step(
                    document_id,
                    "raw_upload",
                    lambda: self._storage.put_bytes(
                        raw_key,
                        content,
                        content_type=inspection.detected_mime_type,
                        metadata={
                            "checksum-sha256": metadata.checksum,
                            "document-id": document_id,
                        },
                    ),
                )
            require_indexable_object(
                inspection,
                stage="text_extraction",
            )
            extraction = self._log.run_step(
                document_id,
                "text_extraction",
                lambda: self._extractors.extract_with_metadata(
                    content,
                    metadata.file_type,
                ),
            )
            metadata_payload: dict[str, object] = {
                "document_id": document_id,
                "metadata": metadata.to_dict(),
                "raw_key": raw_key,
            }
            if extraction is not None and extraction.metadata:
                metadata_payload["extraction"] = extraction.metadata
            self._log.run_step(
                document_id,
                "metadata_upload",
                lambda: self._storage.put_json(
                    KnowledgeKeys.metadata(document_id),
                    metadata_payload,
                ),
            )
            text = extraction.text if extraction is not None else None

            processed_key: str | None = None
            if text is not None:
                processed_key = KnowledgeKeys.processed(document_id)
                encoded_text = text.encode("utf-8")
                self._log.run_step(
                    document_id,
                    "processed_upload",
                    lambda: self._storage.put_bytes(
                        processed_key,
                        encoded_text,
                        content_type="text/plain; charset=utf-8",
                    ),
                )

            require_indexable_object(
                inspection,
                stage="chunk_generation",
            )
            chunks = self._log.run_step(
                document_id,
                "chunking",
                lambda: (
                    self._chunker.chunk(document_id, text)
                    if text is not None
                    else []
                ),
            )
            chunks_key = KnowledgeKeys.chunks(document_id)
            self._log.run_step(
                document_id,
                "chunks_upload",
                lambda: self._storage.put_json(
                    chunks_key,
                    {
                        "document_id": document_id,
                        "chunks": [chunk.to_dict() for chunk in chunks],
                    },
                ),
            )

            embedding_key = KnowledgeKeys.embeddings(document_id)
            embedding_status = EmbeddingStatus.PENDING
            self._log.run_step(
                document_id,
                "embedding_status_upload",
                lambda: self._storage.put_json(
                    embedding_key,
                    self._pending_descriptor(
                        document_id,
                        chunks,
                        inspection=inspection,
                    ),
                ),
            )

            timestamp = metadata.upload_timestamp
            entry = KnowledgeManifestEntry(
                document_id=document_id,
                metadata=metadata,
                chunk_count=len(chunks),
                embedding_status=embedding_status.value,
                ingestion_timestamp=timestamp,
                raw_key=raw_key,
                processed_key=processed_key,
                chunks_key=chunks_key,
                embedding_key=embedding_key,
                pending_chunk_count=len(chunks),
            )
            self._log.run_step(
                document_id,
                "manifest_update",
                lambda: self._manifest.upsert(entry),
            )
        except Exception as error:
            self._log.emit(
                document_id=document_id,
                step="ingestion",
                elapsed_seconds=perf_counter() - ingestion_started,
                success=False,
                error_type=type(error).__name__,
            )
            raise

        self._log.emit(
            document_id=document_id,
            step="ingestion",
            elapsed_seconds=perf_counter() - ingestion_started,
            success=True,
        )
        return entry

    def _pending_descriptor(
        self,
        document_id: str,
        chunks: Sequence[KnowledgeChunk],
        *,
        inspection: ObjectInspection,
    ) -> dict[str, object]:
        """Preserve completed per-chunk state on idempotent re-ingestion."""

        embedding_key = KnowledgeKeys.embeddings(document_id)
        existing = self._storage.get_json(embedding_key)
        expected = {
            chunk.chunk_id: chunk_text_checksum(chunk.text) for chunk in chunks
        }
        raw_states = existing.get("chunks") if existing is not None else None
        if isinstance(raw_states, list):
            stored = {
                state.get("chunk_id"): state.get("checksum")
                for state in raw_states
                if isinstance(state, dict)
            }
            if existing.get("document_id") == document_id and stored == expected:
                return existing
        return {
            "schema_version": 2,
            "document_id": document_id,
            "object_classification": inspection.object_classification.value,
            "indexable": inspection.indexable,
            "storage_only": inspection.storage_only,
            "provider": None,
            "status": EmbeddingStatus.PENDING.value,
            "index_status": "pending",
            "embedding_model": None,
            "vector_store": None,
            "vector_dimension": None,
            "indexed_at": None,
            "chunks": [
                {
                    "attempt_count": 0,
                    "checksum": expected[chunk.chunk_id],
                    "chunk_id": chunk.chunk_id,
                    "indexed_at": None,
                    "last_error_type": None,
                    "last_failure_stage": None,
                    "status": "pending",
                }
                for chunk in chunks
            ],
            "vectors": [],
        }

    def _extract_metadata(
        self,
        *,
        filename: str,
        content: bytes,
        source: str,
        inspection: ObjectInspection,
    ) -> DocumentMetadata:
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        if not filename.strip():
            raise ValueError("filename cannot be empty")
        if PurePath(filename).name != filename or "\\" in filename:
            raise ValueError("filename cannot contain path components")
        if not source.strip():
            raise ValueError("source cannot be empty")
        if len(content) > self._config.maximum_upload_size:
            raise ValueError(
                "document exceeds configured maximum_upload_size"
            )

        suffix = PurePath(filename).suffix.lower().lstrip(".")
        if suffix not in self._config.supported_document_types:
            raise ValueError(f"Unsupported document type '{suffix or 'none'}'")

        timestamp = self._clock()
        if timestamp.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        normalized_timestamp = (
            timestamp.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        return DocumentMetadata(
            filename=filename,
            file_type=suffix,
            upload_timestamp=normalized_timestamp,
            checksum=hashlib.sha256(content).hexdigest(),
            source=source,
            document_size=len(content),
            object_classification=inspection.object_classification.value,
            detected_mime_type=inspection.detected_mime_type,
            declared_mime_type=inspection.declared_mime_type,
            file_extension=inspection.file_extension,
            media_type=(
                inspection.media_type.value
                if inspection.media_type is not None
                else None
            ),
            storage_only=inspection.storage_only,
            indexable=inspection.indexable,
            quarantine_reason=inspection.quarantine_reason,
            source_s3_uri=source if source.startswith("s3://") else None,
            checksum_sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
        )


# Vector indexing is composed at the event/application boundary so extraction,
# chunking, and persistence remain provider-neutral and independently testable.
