"""Provider-neutral ingestion and retrieval for approved bookkeeping PDFs."""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import logging
import re
from time import perf_counter
from typing import Any, Mapping, Sequence

from bookkeeping.knowledge_models import (
    BookkeepingCitation,
    BookkeepingDocumentType,
    BookkeepingKnowledgeDocument,
    BookkeepingKnowledgeMetadata,
    BookkeepingKnowledgeResult,
    BookkeepingMetadataValidationError,
    BookkeepingRetrievalError,
    BookkeepingRetrievedPassage,
    BookkeepingRetrievalMode,
)
from knowledge.chunking import TextChunker
from knowledge.config import KnowledgeConfig
from knowledge.embedding_workflow import EmbeddingWorkflow
from knowledge.embeddings import EmbeddingProvider, EmbeddingStatus
from knowledge.hybrid_retrieval import ReciprocalRankFusionRetriever
from knowledge.ingestion import KnowledgeIngestionPipeline
from knowledge.keyword_retrieval import InMemoryBM25Retriever
from knowledge.manifest import KnowledgeManifestRepository
from knowledge.models import EmbeddingRecord, KnowledgeChunk
from knowledge.pdf_extraction import PDF_PAGE_SEPARATOR
from knowledge.retrieval import (
    InMemoryCosineRetriever,
    RetrievalEntry,
    RetrievalResult,
)
from knowledge.storage import KnowledgeKeys, KnowledgeStorage


logger = logging.getLogger(__name__)

_ACCOUNT_NUMBER_PATTERN = re.compile(
    r"(?i)(?:(?:account|acct|routing|iban)\s*(?:number|no\.?|#)?\s*[:=-]?\s*)"
    r"[a-z0-9][a-z0-9 -]{5,34}"
)
_LONG_DIGIT_PATTERN = re.compile(r"(?<![\w])\d{7,}(?![\w])")
_POLICY_RULE_PATTERN = re.compile(
    r"(?i)\b(?P<subject>[a-z][a-z0-9 &/-]{2,60}?)\s+"
    r"(?:must|should)\s+be\s+"
    r"(?:categorized|classified|recorded)\s+as\s+"
    r"(?P<category>[a-z][a-z &/-]{1,40}?)(?:[.;\n]|$)"
)


class InMemoryKnowledgeStorage:
    """Session-local implementation of the existing knowledge storage API."""

    def __init__(self) -> None:
        self.bytes_by_key: dict[str, bytes] = {}
        self.json_by_key: dict[str, dict[str, Any]] = {}
        self.content_type_by_key: dict[str, str] = {}
        self.metadata_by_key: dict[str, dict[str, str]] = {}

    def put_bytes(
        self,
        key: str,
        content: bytes,
        *,
        content_type: str,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        self.bytes_by_key[key] = bytes(content)
        self.content_type_by_key[key] = content_type
        self.metadata_by_key[key] = dict(metadata or {})

    def put_json(self, key: str, payload: Mapping[str, Any]) -> None:
        # JSON round-tripping prevents callers from mutating stored values.
        self.json_by_key[key] = json.loads(json.dumps(payload))

    def get_json(self, key: str) -> dict[str, Any] | None:
        payload = self.json_by_key.get(key)
        return json.loads(json.dumps(payload)) if payload is not None else None


class _PageAwarePdfChunker:
    """Reuse TextChunker per extracted page so citations never cross pages."""

    def __init__(self, chunk_size: int, overlap: int) -> None:
        self._chunker = TextChunker(chunk_size, overlap)

    def chunk(self, document_id: str, text: str) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        page_start = 0
        for page_text in text.split(PDF_PAGE_SEPARATOR):
            for local_chunk in self._chunker.chunk(document_id, page_text):
                index = len(chunks)
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=f"{document_id}:{index:06d}",
                        document_id=document_id,
                        index=index,
                        text=local_chunk.text,
                        start_character=(
                            page_start + local_chunk.start_character
                        ),
                        end_character=page_start + local_chunk.end_character,
                    )
                )
            page_start += len(page_text) + len(PDF_PAGE_SEPARATOR)
        return chunks


@dataclass(frozen=True)
class _StoredReference:
    document: BookkeepingKnowledgeDocument
    entries: tuple[RetrievalEntry, ...]


class BookkeepingKnowledgeService:
    """Ingest and retrieve only approved, correctly scoped PDF references."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        chunk_size: int = 1_000,
        overlap: int = 100,
        maximum_upload_size: int = 10 * 1024 * 1024,
        maximum_passages: int = 5,
        maximum_context_characters: int = 12_000,
        minimum_semantic_similarity: float = -1.0,
        event_logger: logging.Logger | None = None,
    ) -> None:
        if maximum_passages <= 0:
            raise ValueError("maximum_passages must be greater than zero")
        if maximum_context_characters <= 0:
            raise ValueError(
                "maximum_context_characters must be greater than zero"
            )
        if not -1.0 <= minimum_semantic_similarity <= 1.0:
            raise ValueError(
                "minimum_semantic_similarity must be between -1 and 1"
            )
        self._embedding_provider = embedding_provider
        self._config = KnowledgeConfig(
            chunk_size=chunk_size,
            overlap=overlap,
            supported_document_types=frozenset({"pdf"}),
            maximum_upload_size=maximum_upload_size,
        )
        self._maximum_passages = maximum_passages
        self._maximum_context_characters = maximum_context_characters
        self._minimum_semantic_similarity = minimum_semantic_similarity
        self._storage: KnowledgeStorage = InMemoryKnowledgeStorage()
        self._manifest = KnowledgeManifestRepository(self._storage)
        self._references: dict[str, _StoredReference] = {}
        self._logger = event_logger or logger

    def ingest_pdf(
        self,
        *,
        filename: str,
        content: bytes,
        metadata: BookkeepingKnowledgeMetadata,
    ) -> BookkeepingKnowledgeDocument:
        """Use the existing extraction/chunk/embedding workflow locally."""

        started = perf_counter()
        if not isinstance(metadata, BookkeepingKnowledgeMetadata):
            raise BookkeepingMetadataValidationError(
                "metadata must be BookkeepingKnowledgeMetadata"
            )
        if filename != metadata.source_filename:
            raise BookkeepingMetadataValidationError(
                "filename must match metadata source_filename"
            )
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        document_id = self._document_id(content, metadata)
        try:
            pipeline = KnowledgeIngestionPipeline(
                self._storage,
                self._config,
                chunker=_PageAwarePdfChunker(
                    self._config.chunk_size,
                    self._config.overlap,
                ),
                manifest=self._manifest,
                document_id_factory=lambda: document_id,
                event_logger=self._logger,
            )
            manifest_entry = pipeline.ingest(
                filename=filename,
                content=content,
                source=f"session-upload://{document_id}/{filename}",
            )
            chunks = self._load_chunks(document_id)
            report = EmbeddingWorkflow(
                storage=self._storage,
                provider=self._embedding_provider,
                model_id=self._embedding_model_id(),
                batch_size=max(1, min(16, len(chunks) or 1)),
                manifest=self._manifest,
                event_logger=self._logger,
            ).embed_document(
                document_id=document_id,
                chunks=chunks,
                source_object_key=manifest_entry.chunks_key,
                object_classification=(
                    manifest_entry.metadata.object_classification
                ),
            )
            embedding_status = (
                EmbeddingStatus.COMPLETE
                if report.succeeded
                else EmbeddingStatus.FAILED
            )
            manifest_entry = replace(
                manifest_entry,
                embedding_status=embedding_status.value,
            )
            self._manifest.upsert(manifest_entry)
            extraction = self._storage.get_json(
                KnowledgeKeys.metadata(document_id)
            ) or {}
            extraction_metadata = extraction.get("extraction", {})
            if not isinstance(extraction_metadata, dict):
                extraction_metadata = {}
            processed = self._processed_text(document_id)
            page_by_chunk = {
                chunk.chunk_id: _page_for_offset(
                    processed,
                    chunk.start_character,
                )
                for chunk in chunks
            }
            records = {
                record.chunk_id: record for record in report.created
            }
            entries = tuple(
                self._retrieval_entry(
                    chunk,
                    records[chunk.chunk_id],
                    metadata,
                    page_by_chunk[chunk.chunk_id],
                )
                for chunk in chunks
                if chunk.chunk_id in records
            )
            document = BookkeepingKnowledgeDocument(
                manifest_entry=manifest_entry,
                classification=metadata,
                source_metadata=manifest_entry.metadata,
                extraction_metadata=extraction_metadata,
                page_by_chunk_id=page_by_chunk,
            )
            self._references[document_id] = _StoredReference(
                document=document,
                entries=entries,
            )
        except Exception as error:
            self._emit(
                event="bookkeeping_knowledge_ingestion",
                success=False,
                elapsed_seconds=perf_counter() - started,
                document_id=document_id,
                client_id=metadata.client_id,
                error_type=type(error).__name__,
            )
            raise

        self._emit(
            event="bookkeeping_knowledge_ingestion",
            success=True,
            elapsed_seconds=perf_counter() - started,
            document_id=document_id,
            client_id=metadata.client_id,
            chunk_count=len(entries),
        )
        return document

    def deactivate(self, document_id: str) -> BookkeepingKnowledgeDocument:
        """Exclude a reference from retrieval without deleting its content."""

        stored = self._references.get(document_id)
        if stored is None:
            raise BookkeepingRetrievalError("Reference was not found")
        document = replace(
            stored.document,
            classification=stored.document.classification.deactivate(),
        )
        self._references[document_id] = replace(stored, document=document)
        return document

    def list_documents(
        self,
        *,
        client_id: str | None = None,
        include_inactive: bool = True,
    ) -> tuple[BookkeepingKnowledgeDocument, ...]:
        """List only general references and the requested client's references."""

        scope = _normalize_optional_client_id(client_id)
        visible = []
        for stored in self._references.values():
            metadata = stored.document.classification
            if metadata.client_specific and metadata.client_id != scope:
                continue
            if not include_inactive and not metadata.active:
                continue
            visible.append(stored.document)
        return tuple(
            sorted(
                visible,
                key=lambda item: (
                    item.classification.title.casefold(),
                    item.document_id,
                ),
            )
        )

    def search(
        self,
        question: str,
        *,
        client_id: str | None = None,
        retrieval_mode: BookkeepingRetrievalMode | str = (
            BookkeepingRetrievalMode.KEYWORD
        ),
        maximum_passages: int | None = None,
        maximum_context_characters: int | None = None,
        document_types: Sequence[BookkeepingDocumentType | str] | None = None,
    ) -> BookkeepingKnowledgeResult:
        """Retrieve bounded passages from approved and isolated references."""

        started = perf_counter()
        query = _validate_question(question)
        scope = _normalize_optional_client_id(client_id)
        try:
            mode = BookkeepingRetrievalMode(retrieval_mode)
        except ValueError as error:
            raise BookkeepingRetrievalError(
                "retrieval_mode must be keyword, semantic, or hybrid"
            ) from error
        passage_limit = maximum_passages or self._maximum_passages
        context_limit = (
            maximum_context_characters
            or self._maximum_context_characters
        )
        if passage_limit <= 0 or passage_limit > self._maximum_passages:
            raise BookkeepingRetrievalError(
                "maximum_passages exceeds the configured limit"
            )
        if context_limit <= 0 or context_limit > self._maximum_context_characters:
            raise BookkeepingRetrievalError(
                "maximum_context_characters exceeds the configured limit"
            )
        allowed_types = _normalize_document_types(document_types)
        eligible = self._eligible_entries(scope, allowed_types)
        if not eligible:
            return self._no_context(query, mode)

        retrieval_scope = scope or "general-reference"
        scoped_entries = tuple(
            _entry_for_retriever(entry, retrieval_scope)
            for entry in eligible
        )
        try:
            results = self._retrieve(
                scoped_entries,
                query,
                mode,
                retrieval_scope,
                passage_limit,
            )
        except Exception as error:
            self._emit(
                event="bookkeeping_knowledge_retrieval",
                success=False,
                elapsed_seconds=perf_counter() - started,
                client_id=scope,
                retrieval_mode=mode.value,
                error_type=type(error).__name__,
            )
            if isinstance(error, BookkeepingRetrievalError):
                raise
            raise BookkeepingRetrievalError(
                "Approved bookkeeping references could not be searched"
            ) from error

        passages = self._bounded_passages(
            results,
            context_limit=context_limit,
            passage_limit=passage_limit,
        )
        conflicts = _detect_conflicts(passages)
        warnings: list[str] = []
        if not passages:
            warnings.append(
                "No relevant approved bookkeeping context was found."
            )
        if conflicts:
            warnings.append(
                "Retrieved policy passages conflict and require human review."
            )
        outcome = BookkeepingKnowledgeResult(
            question=query,
            retrieval_mode=mode,
            passages=passages,
            relevant_context_found=bool(passages),
            sources_conflict=bool(conflicts),
            conflict_details=conflicts,
            warnings=tuple(warnings),
        )
        self._emit(
            event="bookkeeping_knowledge_retrieval",
            success=True,
            elapsed_seconds=perf_counter() - started,
            client_id=scope,
            retrieval_mode=mode.value,
            chunk_count=len(passages),
        )
        return outcome

    def _retrieve(
        self,
        entries: tuple[RetrievalEntry, ...],
        query: str,
        mode: BookkeepingRetrievalMode,
        scope: str,
        limit: int,
    ) -> list[RetrievalResult]:
        if mode == BookkeepingRetrievalMode.KEYWORD:
            return InMemoryBM25Retriever(entries).retrieve(
                query,
                client_id=scope,
                environment="bookkeeping",
                top_k=limit,
            )
        query_vectors = self._embedding_provider.embed([query])
        if len(query_vectors) != 1:
            raise BookkeepingRetrievalError(
                "Embedding provider returned an invalid query vector"
            )
        if mode == BookkeepingRetrievalMode.SEMANTIC:
            return InMemoryCosineRetriever(
                entries,
                top_k=limit,
                minimum_similarity=self._minimum_semantic_similarity,
            ).retrieve(query_vectors[0])
        return ReciprocalRankFusionRetriever(entries).retrieve(
            query_vectors[0],
            query,
            client_id=scope,
            environment="bookkeeping",
            top_k=limit,
        )

    def _eligible_entries(
        self,
        client_id: str | None,
        allowed_types: frozenset[BookkeepingDocumentType] | None,
    ) -> tuple[RetrievalEntry, ...]:
        entries: list[RetrievalEntry] = []
        for stored in self._references.values():
            metadata = stored.document.classification
            if (
                metadata.domain != "bookkeeping"
                or not metadata.approved_for_bookkeeping
                or not metadata.active
                or (
                    allowed_types is not None
                    and metadata.document_type not in allowed_types
                )
            ):
                continue
            if metadata.client_specific:
                if client_id is None or metadata.client_id != client_id:
                    continue
            elif metadata.client_id is not None:
                # Defensive fail-closed check even though validation rejects it.
                continue
            entries.extend(stored.entries)
        return tuple(entries)

    def _bounded_passages(
        self,
        results: Sequence[RetrievalResult],
        *,
        context_limit: int,
        passage_limit: int,
    ) -> tuple[BookkeepingRetrievedPassage, ...]:
        passages: list[BookkeepingRetrievedPassage] = []
        remaining = context_limit
        for result in results:
            if len(passages) >= passage_limit or remaining <= 0:
                break
            metadata = dict(result.metadata)
            text = redact_bookkeeping_text(result.text).strip()
            if not text:
                continue
            bounded_text = text[:remaining]
            if not bounded_text.strip():
                break
            citation_id = str(len(passages) + 1)
            page = metadata.get("page_number")
            if not isinstance(page, int) or isinstance(page, bool) or page <= 0:
                page = None
            citation = BookkeepingCitation(
                citation_id=citation_id,
                document_id=result.document_id,
                document_title=str(metadata["title"]),
                source_filename=str(metadata["source_filename"]),
                page_number=page,
                chunk_id=result.chunk_id,
                retrieval_score=result.similarity_score,
            )
            passages.append(
                BookkeepingRetrievedPassage(
                    text=bounded_text,
                    citation=citation,
                    document_type=BookkeepingDocumentType(
                        metadata["document_type"]
                    ),
                    client_id=(
                        str(metadata["reference_client_id"])
                        if metadata.get("reference_client_id") is not None
                        else None
                    ),
                )
            )
            remaining -= len(bounded_text)
        return tuple(passages)

    def _retrieval_entry(
        self,
        chunk: KnowledgeChunk,
        record: EmbeddingRecord,
        metadata: BookkeepingKnowledgeMetadata,
        page_number: int | None,
    ) -> RetrievalEntry:
        return RetrievalEntry(
            embedding_record=record,
            source=metadata.title,
            text=chunk.text,
            metadata={
                **metadata.to_dict(),
                "reference_client_id": metadata.client_id,
                "page_number": page_number,
                "object_key": record.source_object_key,
                "environment": "bookkeeping",
                "object_classification": "indexable_text_document",
                "indexable": True,
                "storage_only": False,
            },
        )

    def _load_chunks(self, document_id: str) -> tuple[KnowledgeChunk, ...]:
        payload = self._storage.get_json(KnowledgeKeys.chunks(document_id))
        items = payload.get("chunks") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise BookkeepingRetrievalError("Ingested chunks are unavailable")
        try:
            return tuple(
                KnowledgeChunk(
                    chunk_id=str(item["chunk_id"]),
                    document_id=str(item["document_id"]),
                    index=int(item["index"]),
                    text=str(item["text"]),
                    start_character=int(item["start_character"]),
                    end_character=int(item["end_character"]),
                )
                for item in items
                if isinstance(item, dict)
            )
        except (KeyError, TypeError, ValueError) as error:
            raise BookkeepingRetrievalError(
                "Ingested chunks are malformed"
            ) from error

    def _processed_text(self, document_id: str) -> str:
        storage = self._storage
        if not isinstance(storage, InMemoryKnowledgeStorage):
            raise BookkeepingRetrievalError(
                "Session-local processed text is unavailable"
            )
        content = storage.bytes_by_key.get(KnowledgeKeys.processed(document_id))
        if content is None:
            raise BookkeepingRetrievalError("Processed PDF text is unavailable")
        return content.decode("utf-8")

    def _embedding_model_id(self) -> str:
        value = getattr(self._embedding_provider, "model_id", None)
        if not isinstance(value, str) or not value.strip():
            return f"{self._embedding_provider.provider_name}-bookkeeping"
        return value.strip()

    @staticmethod
    def _document_id(
        content: bytes,
        metadata: BookkeepingKnowledgeMetadata,
    ) -> str:
        classification = json.dumps(metadata.to_dict(), sort_keys=True)
        digest = hashlib.sha256()
        digest.update(content)
        digest.update(b"\0")
        digest.update(classification.encode("utf-8"))
        return digest.hexdigest()[:32]

    @staticmethod
    def _no_context(
        question: str,
        mode: BookkeepingRetrievalMode,
    ) -> BookkeepingKnowledgeResult:
        return BookkeepingKnowledgeResult(
            question=question,
            retrieval_mode=mode,
            passages=(),
            relevant_context_found=False,
            warnings=(
                "No relevant approved bookkeeping context was found.",
            ),
        )

    def _emit(
        self,
        *,
        event: str,
        success: bool,
        elapsed_seconds: float,
        document_id: str | None = None,
        client_id: str | None = None,
        retrieval_mode: str | None = None,
        chunk_count: int | None = None,
        error_type: str | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "elapsed_ms": round(elapsed_seconds * 1_000, 3),
            "event": event,
            "success": success,
        }
        for key, value in (
            ("document_id", document_id),
            ("client_id", client_id),
            ("retrieval_mode", retrieval_mode),
            ("chunk_count", chunk_count),
            ("error_type", error_type),
        ):
            if value is not None:
                payload[key] = value
        self._logger.info(json.dumps(payload, sort_keys=True))


def format_bookkeeping_citation(citation: BookkeepingCitation) -> str:
    """Render only verified, available fields in the repository convention."""

    parts = [f"[{citation.citation_id}] {citation.document_title}"]
    if citation.page_number is not None:
        parts.append(f"page {citation.page_number}")
    parts.append(f"chunk {citation.chunk_id}")
    return " — ".join(parts)


def _entry_for_retriever(
    entry: RetrievalEntry,
    client_scope: str,
) -> RetrievalEntry:
    metadata = dict(entry.metadata)
    metadata["client_id"] = client_scope
    metadata["environment"] = "bookkeeping"
    return RetrievalEntry(
        embedding_record=entry.embedding_record,
        source=entry.source,
        text=entry.text,
        metadata=metadata,
    )


def _page_for_offset(text: str, offset: int) -> int | None:
    if not text or offset < 0 or offset > len(text):
        return None
    return text.count(PDF_PAGE_SEPARATOR, 0, offset) + 1


def redact_bookkeeping_text(text: str) -> str:
    """Remove likely raw account identifiers from display/model text."""

    redacted = _ACCOUNT_NUMBER_PATTERN.sub("[redacted account identifier]", text)
    return _LONG_DIGIT_PATTERN.sub("[redacted number]", redacted)


def _detect_conflicts(
    passages: Sequence[BookkeepingRetrievedPassage],
) -> tuple[str, ...]:
    rules: dict[str, dict[str, set[str]]] = {}
    for passage in passages:
        for match in _POLICY_RULE_PATTERN.finditer(passage.text):
            subject = " ".join(match.group("subject").casefold().split())
            category = " ".join(match.group("category").casefold().split())
            rules.setdefault(subject, {}).setdefault(category, set()).add(
                passage.citation.citation_id
            )
    conflicts = []
    for subject, categories in sorted(rules.items()):
        if len(categories) <= 1:
            continue
        category_text = ", ".join(sorted(categories))
        citation_ids = sorted(
            {item for values in categories.values() for item in values}
        )
        conflicts.append(
            f"Conflicting guidance for '{subject}': {category_text} "
            f"(citations {', '.join(citation_ids)})."
        )
    return tuple(conflicts)


def _normalize_optional_client_id(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip().casefold()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", normalized):
        raise BookkeepingRetrievalError(
            "client_id contains unsupported characters"
        )
    return normalized


def _normalize_document_types(
    values: Sequence[BookkeepingDocumentType | str] | None,
) -> frozenset[BookkeepingDocumentType] | None:
    if values is None:
        return None
    try:
        parsed = frozenset(BookkeepingDocumentType(value) for value in values)
    except ValueError as error:
        raise BookkeepingRetrievalError(
            "document_types contains an unsupported value"
        ) from error
    if not parsed:
        raise BookkeepingRetrievalError("document_types cannot be empty")
    return parsed


def _validate_question(value: str) -> str:
    if not isinstance(value, str):
        raise BookkeepingRetrievalError("question must be a string")
    normalized = value.strip()
    if not normalized:
        raise BookkeepingRetrievalError("question cannot be empty")
    if len(normalized) > 2_000:
        raise BookkeepingRetrievalError("question exceeds 2,000 characters")
    return normalized
