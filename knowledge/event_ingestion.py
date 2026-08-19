"""Safe orchestration for S3 ObjectCreated knowledge-ingestion events."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import PurePosixPath
import re
from time import perf_counter
from typing import Any, Protocol
from urllib.parse import quote, unquote_plus

from knowledge.config import KnowledgeConfig
from knowledge.extraction import EXTRACTABLE_DOCUMENT_TYPES
from knowledge.ingestion import KnowledgeIngestionPipeline
from knowledge.manifest import KnowledgeManifestRepository
from knowledge.media_classification import (
    MediaType,
    ObjectClassification,
    ObjectInspection,
    classify_uploaded_object,
    require_indexable_object,
)
from knowledge.models import KnowledgeManifestEntry
from knowledge.storage import KnowledgeKeys, KnowledgeStorage
from knowledge.vector_indexing import VectorIndexingReport


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionEventSummary:
    """Stable counters returned for one S3 event payload."""

    records_received: int = 0
    records_processed: int = 0
    records_skipped: int = 0
    records_failed: int = 0
    indexable_documents_received: int = 0
    media_objects_stored: int = 0
    unsupported_binaries_stored: int = 0
    suspicious_objects_quarantined: int = 0
    media_indexing_attempts_blocked: int = 0
    mime_mismatch_count: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "records_received": self.records_received,
            "records_processed": self.records_processed,
            "records_skipped": self.records_skipped,
            "records_failed": self.records_failed,
            "indexable_documents_received": (
                self.indexable_documents_received
            ),
            "media_objects_stored": self.media_objects_stored,
            "unsupported_binaries_stored": (
                self.unsupported_binaries_stored
            ),
            "suspicious_objects_quarantined": (
                self.suspicious_objects_quarantined
            ),
            "media_indexing_attempts_blocked": (
                self.media_indexing_attempts_blocked
            ),
            "mime_mismatch_count": self.mime_mismatch_count,
        }


@dataclass(frozen=True)
class _RecordResult:
    outcome: str
    inspection: ObjectInspection | None = None


class IngestionBatchError(RuntimeError):
    """Raised after all practical records run when any valid record failed."""

    def __init__(self, summary: IngestionEventSummary) -> None:
        self.summary = summary.to_dict()
        super().__init__(
            "S3 knowledge ingestion completed with "
            f"{summary.records_failed} failed record(s)"
        )


class AutomaticIndexingIncompleteError(RuntimeError):
    """Signal the event retry path after partial state has been persisted."""

    def __init__(self, report: VectorIndexingReport) -> None:
        self.document_id = report.document_id
        self.index_status = report.vector_status.value
        self.failed_chunk_count = report.statistics.failed_chunk_count
        super().__init__(
            "Automatic vector indexing remains incomplete for "
            f"document '{report.document_id}'"
        )


class PendingDocumentIndexer(Protocol):
    """Dependency-injection boundary for automatic post-ingestion indexing."""

    def index_pending_document(
        self,
        entry: KnowledgeManifestEntry,
        *,
        client_id: str,
        environment: str,
        knowledge_namespace: str | None = None,
        knowledge_domain: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VectorIndexingReport:
        """Advance one persisted descriptor."""


class S3DocumentIngestionProcessor:
    """Validate S3 records and delegate document processing to the pipeline."""

    OUTPUT_PREFIXES = (
        f"{KnowledgeKeys.PROCESSED}/",
        f"{KnowledgeKeys.CHUNKS}/",
        f"{KnowledgeKeys.EMBEDDINGS}/",
        f"{KnowledgeKeys.METADATA}/",
        f"{KnowledgeKeys.MEDIA}/",
        f"{KnowledgeKeys.QUARANTINE}/",
    )

    def __init__(
        self,
        *,
        bucket_name: str,
        raw_prefix: str,
        s3_client: Any,
        storage: KnowledgeStorage,
        pipeline: KnowledgeIngestionPipeline,
        manifest: KnowledgeManifestRepository,
        config: KnowledgeConfig,
        client_id: str,
        environment: str,
        indexing_service: PendingDocumentIndexer | None = None,
        knowledge_namespace: str = "default",
        knowledge_domain: str = "general",
        event_logger: logging.Logger | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        normalized_bucket = bucket_name.strip()
        normalized_prefix = raw_prefix.strip().lstrip("/")
        if not normalized_bucket:
            raise ValueError("bucket_name cannot be empty")
        if not normalized_prefix or not normalized_prefix.endswith("/"):
            raise ValueError(
                "raw_prefix must be a non-empty prefix ending in '/'"
            )
        if not client_id.strip():
            raise ValueError("client_id cannot be empty")
        if not environment.strip():
            raise ValueError("environment cannot be empty")
        if not knowledge_namespace.strip():
            raise ValueError("knowledge_namespace cannot be empty")
        if not knowledge_domain.strip():
            raise ValueError("knowledge_domain cannot be empty")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", client_id.strip()):
            raise ValueError("client_id must be safe for an S3 key segment")
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]*", environment.strip()
        ):
            raise ValueError("environment must be safe for an S3 key segment")
        unsupported_extractors = (
            config.supported_document_types - EXTRACTABLE_DOCUMENT_TYPES
        )
        if unsupported_extractors:
            unsupported = ", ".join(sorted(unsupported_extractors))
            raise ValueError(
                "event ingestion requires registered text extractors; "
                f"unsupported configured types: {unsupported}"
            )

        self._bucket_name = normalized_bucket
        self._raw_prefix = normalized_prefix
        self._s3_client = s3_client
        self._storage = storage
        self._pipeline = pipeline
        self._manifest = manifest
        self._config = config
        self._client_id = client_id.strip()
        self._environment = environment.strip()
        self._indexing_service = indexing_service
        self._knowledge_namespace = knowledge_namespace.strip()
        self._knowledge_domain = knowledge_domain.strip()
        self._logger = event_logger or logger
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def process_event(self, event: object) -> dict[str, int]:
        """Process every practical record, then raise if a valid record failed."""

        records = event.get("Records") if isinstance(event, dict) else None
        if not isinstance(records, list):
            summary = IngestionEventSummary()
            self._emit(
                record_index=None,
                outcome="skipped",
                reason="malformed_event",
            )
            return summary.to_dict()

        processed = 0
        skipped = 0
        failures: list[Exception] = []
        counters = {
            "indexable_documents_received": 0,
            "media_objects_stored": 0,
            "unsupported_binaries_stored": 0,
            "suspicious_objects_quarantined": 0,
            "media_indexing_attempts_blocked": 0,
            "mime_mismatch_count": 0,
        }
        for index, record in enumerate(records):
            started = perf_counter()
            try:
                result = self._process_record(record, index)
            except Exception as error:
                failures.append(error)
                self._emit(
                    record_index=index,
                    outcome="failed",
                    reason="record_processing_failed",
                    elapsed_seconds=perf_counter() - started,
                    error_type=type(error).__name__,
                )
                continue

            if result.outcome == "processed":
                processed += 1
            else:
                skipped += 1
            if result.inspection is not None:
                classification = result.inspection.object_classification
                if classification is ObjectClassification.INDEXABLE_TEXT_DOCUMENT:
                    counters["indexable_documents_received"] += 1
                else:
                    counters["media_indexing_attempts_blocked"] += 1
                if classification is ObjectClassification.MEDIA_OBJECT:
                    counters["media_objects_stored"] += 1
                elif classification is ObjectClassification.UNSUPPORTED_BINARY:
                    counters["unsupported_binaries_stored"] += 1
                elif classification is ObjectClassification.REJECTED_OR_SUSPICIOUS:
                    counters["suspicious_objects_quarantined"] += 1
                if (
                    result.inspection.quarantine_reason
                    == "declared_mime_type_mismatch"
                ):
                    counters["mime_mismatch_count"] += 1

        summary = IngestionEventSummary(
            records_received=len(records),
            records_processed=processed,
            records_skipped=skipped,
            records_failed=len(failures),
            **counters,
        )
        self._emit(
            record_index=None,
            outcome="failed" if failures else "complete",
            reason="batch_summary",
            summary=summary,
        )
        if failures:
            raise IngestionBatchError(summary) from failures[0]
        return summary.to_dict()

    def _process_record(self, record: object, index: int) -> _RecordResult:
        started = perf_counter()
        parsed = self._parse_record(record)
        if isinstance(parsed, str):
            self._emit(
                record_index=index,
                outcome="skipped",
                reason=parsed,
                elapsed_seconds=perf_counter() - started,
            )
            return _RecordResult("skipped")

        key = parsed["key"]
        file_type = PurePosixPath(key).suffix.lower().lstrip(".")

        get_parameters: dict[str, str] = {
            "Bucket": self._bucket_name,
            "Key": key,
        }
        if parsed["version_id"] is not None:
            get_parameters["VersionId"] = parsed["version_id"]
        response = self._s3_client.get_object(**get_parameters)
        body = response.get("Body")
        if body is None or not hasattr(body, "read"):
            raise ValueError("S3 response Body must provide read()")
        content, size_bytes, checksum = self._read_body(body)
        declared_mime_type = self._optional_string(response.get("ContentType"))
        filename = PurePosixPath(key).name
        inspection = classify_uploaded_object(
            filename=filename,
            declared_mime_type=declared_mime_type,
            content=content,
            supported_document_types=self._config.supported_document_types,
        )
        if (
            inspection.indexable
            and size_bytes > self._config.maximum_upload_size
        ):
            inspection = ObjectInspection(
                object_classification=(
                    ObjectClassification.REJECTED_OR_SUSPICIOUS
                ),
                detected_mime_type=inspection.detected_mime_type,
                declared_mime_type=inspection.declared_mime_type,
                file_extension=inspection.file_extension,
                media_type=None,
                storage_only=True,
                indexable=False,
                quarantine_reason="maximum_upload_size_exceeded",
            )
        version_id = self._optional_string(
            response.get("VersionId")
        ) or parsed["version_id"]
        etag = self._normalize_etag(
            response.get("ETag")
        ) or parsed["etag"]
        trusted_document_id = self._trusted_pipeline_document_id(
            response.get("Metadata"),
            checksum,
        )
        document_id = trusted_document_id or self._document_id(
            key=key,
            version_id=version_id,
            etag=etag,
            checksum=checksum,
        )

        source = self._source_uri(key, version_id)
        if not inspection.indexable:
            self._store_non_indexable(
                object_id=document_id,
                filename=filename,
                source_key=key,
                source_uri=source,
                source_version_id=version_id,
                inspection=inspection,
                checksum=checksum,
                size_bytes=size_bytes,
                upload_timestamp=self._upload_timestamp(
                    parsed.get("event_time"),
                    response.get("LastModified"),
                ),
            )
            quarantined = (
                inspection.object_classification
                is ObjectClassification.REJECTED_OR_SUSPICIOUS
            )
            self._emit(
                record_index=index,
                outcome="processed",
                reason=(
                    "suspicious_object_quarantined"
                    if quarantined
                    else "storage_only_object_stored"
                ),
                elapsed_seconds=perf_counter() - started,
                document_id=document_id,
                file_type=file_type or "none",
                key_digest=self._key_digest(key),
                inspection=inspection,
                warning=quarantined,
            )
            return _RecordResult("processed", inspection)

        require_indexable_object(inspection, stage="ingestion_routing")

        existing = self._manifest.get(document_id)
        if existing is not None:
            metadata = existing.get("metadata")
            if (
                isinstance(metadata, dict)
                and metadata.get("checksum") == checksum
                and existing.get("raw_key") == key
            ):
                if (
                    self._indexing_service is not None
                    and existing.get("index_status") != "complete"
                ):
                    self._index_entry(KnowledgeManifestEntry.from_dict(existing))
                    self._emit(
                        record_index=index,
                        outcome="processed",
                        reason="indexing_retry_complete",
                        elapsed_seconds=perf_counter() - started,
                        document_id=document_id,
                        file_type=file_type,
                        key_digest=self._key_digest(key),
                    )
                    return _RecordResult("processed", inspection)
                self._emit(
                    record_index=index,
                    outcome="skipped",
                    reason="already_processed",
                    elapsed_seconds=perf_counter() - started,
                    document_id=document_id,
                    file_type=file_type,
                    key_digest=self._key_digest(key),
                )
                return _RecordResult("skipped", inspection)
            raise ValueError(
                "Existing manifest entry conflicts with source identity"
            )

        entry = self._pipeline.ingest_existing_raw(
            document_id=document_id,
            filename=filename,
            content=content,
            source=source,
            raw_key=key,
            inspection=inspection,
            declared_mime_type=declared_mime_type,
        )
        if self._indexing_service is not None:
            self._index_entry(entry)
        self._emit(
            record_index=index,
            outcome="processed",
            reason="ingestion_complete",
            elapsed_seconds=perf_counter() - started,
            document_id=document_id,
            file_type=file_type,
            key_digest=self._key_digest(key),
        )
        return _RecordResult("processed", inspection)

    def _index_entry(self, entry: KnowledgeManifestEntry) -> None:
        if self._indexing_service is None:
            return
        require_indexable_object(
            entry.metadata.object_classification,
            stage="automatic_indexing",
        )
        report = self._indexing_service.index_pending_document(
            entry,
            client_id=self._client_id,
            environment=self._environment,
            knowledge_namespace=self._knowledge_namespace,
            knowledge_domain=self._knowledge_domain,
        )
        if not report.succeeded:
            raise AutomaticIndexingIncompleteError(report)

    def _read_body(self, body: Any) -> tuple[bytes, int, str]:
        """Hash the complete body while bounding bytes retained for text work."""

        retained = bytearray()
        digest = hashlib.sha256()
        size_bytes = 0
        retain_limit = self._config.maximum_upload_size + 1
        while True:
            chunk = body.read(64 * 1024)
            if not isinstance(chunk, bytes):
                raise TypeError("S3 response Body must return bytes")
            if not chunk:
                break
            digest.update(chunk)
            size_bytes += len(chunk)
            if len(retained) < retain_limit:
                remaining = retain_limit - len(retained)
                retained.extend(chunk[:remaining])
        return bytes(retained), size_bytes, digest.hexdigest()

    def _store_non_indexable(
        self,
        *,
        object_id: str,
        filename: str,
        source_key: str,
        source_uri: str,
        source_version_id: str | None,
        inspection: ObjectInspection,
        checksum: str,
        size_bytes: int,
        upload_timestamp: str,
    ) -> None:
        require_indexable = inspection.indexable or not inspection.storage_only
        if require_indexable:
            raise ValueError("storage-only routing requires a non-indexable object")
        if (
            inspection.object_classification
            is ObjectClassification.REJECTED_OR_SUSPICIOUS
        ):
            storage_key = KnowledgeKeys.quarantine(
                self._client_id,
                self._environment,
                object_id,
                filename,
            )
        else:
            media_type = inspection.media_type or MediaType.OTHER
            storage_key = KnowledgeKeys.media(
                self._client_id,
                self._environment,
                media_type.value,
                object_id,
                filename,
            )
        copy_source: dict[str, str] = {
            "Bucket": self._bucket_name,
            "Key": source_key,
        }
        if source_version_id is not None:
            copy_source["VersionId"] = source_version_id
        self._s3_client.copy_object(
            Bucket=self._bucket_name,
            Key=storage_key,
            CopySource=copy_source,
            MetadataDirective="COPY",
        )
        metadata = {
            "object_id": object_id,
            "original_filename": filename,
            **inspection.to_dict(),
            "source_object_key": source_key,
            "source_s3_uri": source_uri,
            "storage_object_key": storage_key,
            "storage_s3_uri": self._source_uri(storage_key, None),
            "checksum_sha256": checksum,
            "size_bytes": size_bytes,
            "upload_timestamp": upload_timestamp,
            "client_id": self._client_id,
            "environment": self._environment,
        }
        self._storage.put_json(
            KnowledgeKeys.storage_only_metadata(object_id),
            metadata,
        )

    def _upload_timestamp(
        self,
        event_time: object,
        last_modified: object,
    ) -> str:
        if isinstance(event_time, str) and event_time.strip():
            try:
                parsed = datetime.fromisoformat(
                    event_time.strip().replace("Z", "+00:00")
                )
            except ValueError:
                parsed = None
            if parsed is not None and parsed.tzinfo is not None:
                return (
                    parsed.astimezone(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
        if isinstance(last_modified, datetime) and last_modified.tzinfo is not None:
            return (
                last_modified.astimezone(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
        timestamp = self._clock()
        if timestamp.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return (
            timestamp.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    def _parse_record(self, record: object) -> dict[str, Any] | str:
        if not isinstance(record, dict):
            return "malformed_record"
        if record.get("eventSource") != "aws:s3":
            return "non_s3_record"
        event_name = record.get("eventName")
        if (
            not isinstance(event_name, str)
            or not event_name.startswith("ObjectCreated:")
        ):
            return "non_object_created_event"

        s3_details = record.get("s3")
        if not isinstance(s3_details, dict):
            return "malformed_record"
        bucket = s3_details.get("bucket")
        object_details = s3_details.get("object")
        if not isinstance(bucket, dict) or not isinstance(object_details, dict):
            return "malformed_record"
        bucket_name = bucket.get("name")
        encoded_key = object_details.get("key")
        if not isinstance(bucket_name, str) or not isinstance(encoded_key, str):
            return "malformed_record"
        if bucket_name != self._bucket_name:
            return "unexpected_bucket"

        key = unquote_plus(encoded_key)
        if not key or key.endswith("/"):
            return "folder_placeholder"
        if any(key.startswith(prefix) for prefix in self.OUTPUT_PREFIXES):
            return "generated_output"
        if not key.startswith(self._raw_prefix):
            return "outside_raw_prefix"
        if not PurePosixPath(key).name:
            return "folder_placeholder"

        size = object_details.get("size")
        if (
            isinstance(size, bool)
            or (size is not None and not isinstance(size, int))
            or (isinstance(size, int) and size < 0)
        ):
            return "malformed_record"
        return {
            "key": key,
            "size": size,
            "version_id": self._optional_string(
                object_details.get("versionId")
            ),
            "etag": self._normalize_etag(object_details.get("eTag")),
            "event_time": self._optional_string(record.get("eventTime")),
        }

    def _document_id(
        self,
        *,
        key: str,
        version_id: str | None,
        etag: str | None,
        checksum: str,
    ) -> str:
        revision = version_id or etag or "content-sha256"
        identity = "\0".join(
            (
                self._bucket_name,
                key,
                revision,
                checksum,
            )
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _source_uri(self, key: str, version_id: str | None) -> str:
        uri = f"s3://{self._bucket_name}/{quote(key, safe='/')}"
        if version_id is not None:
            uri = f"{uri}?versionId={quote(version_id, safe='')}"
        return uri

    @staticmethod
    def _trusted_pipeline_document_id(
        metadata: object,
        checksum: str,
    ) -> str | None:
        if not isinstance(metadata, dict):
            return None
        document_id = metadata.get("document-id")
        stored_checksum = metadata.get("checksum-sha256")
        if (
            isinstance(document_id, str)
            and re.fullmatch(r"[a-f0-9]{32}", document_id)
            and stored_checksum == checksum
        ):
            return document_id
        return None

    @staticmethod
    def _key_digest(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _optional_string(value: object) -> str | None:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    @classmethod
    def _normalize_etag(cls, value: object) -> str | None:
        normalized = cls._optional_string(value)
        return normalized.strip('"') if normalized is not None else None

    def _emit(
        self,
        *,
        record_index: int | None,
        outcome: str,
        reason: str,
        elapsed_seconds: float | None = None,
        document_id: str | None = None,
        file_type: str | None = None,
        key_digest: str | None = None,
        error_type: str | None = None,
        summary: IngestionEventSummary | None = None,
        inspection: ObjectInspection | None = None,
        warning: bool = False,
    ) -> None:
        event: dict[str, object] = {
            "client_id": self._client_id,
            "environment": self._environment,
            "event": "s3_knowledge_ingestion",
            "outcome": outcome,
            "reason": reason,
        }
        if record_index is not None:
            event["record_index"] = record_index
        if elapsed_seconds is not None:
            event["elapsed_ms"] = round(elapsed_seconds * 1_000, 3)
        if document_id is not None:
            event["document_id"] = document_id
        if file_type is not None:
            event["file_type"] = file_type
        if key_digest is not None:
            event["object_key_digest"] = key_digest
        if error_type is not None:
            event["error_type"] = error_type
        if summary is not None:
            event.update(summary.to_dict())
        if inspection is not None:
            event.update(inspection.to_dict())
        log_method = self._logger.warning if warning else self._logger.info
        log_method(json.dumps(event, sort_keys=True))
