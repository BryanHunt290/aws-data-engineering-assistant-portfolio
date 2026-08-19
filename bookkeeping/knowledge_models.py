"""Typed, privacy-aware models for bookkeeping reference retrieval."""

from dataclasses import dataclass, field, replace
from datetime import date
from enum import StrEnum
import math
from pathlib import PurePath
import re
from types import MappingProxyType
from typing import Any, Mapping

from knowledge.models import DocumentMetadata, KnowledgeManifestEntry


class BookkeepingKnowledgeError(RuntimeError):
    """Base class for safe bookkeeping-knowledge failures."""


class BookkeepingMetadataValidationError(
    BookkeepingKnowledgeError,
    ValueError,
):
    """Raised when bookkeeping reference metadata is malformed or unsafe."""


class BookkeepingRetrievalError(BookkeepingKnowledgeError):
    """Raised when an approved retrieval request cannot be completed."""


class BookkeepingCitationValidationError(BookkeepingKnowledgeError):
    """Raised when generated text cites a passage that was not retrieved."""


class BookkeepingDocumentType(StrEnum):
    """Supported bookkeeping reference classifications."""

    ACCOUNTING_REFERENCE = "accounting_reference"
    BOOKKEEPING_PROCEDURE = "bookkeeping_procedure"
    CHART_OF_ACCOUNTS = "chart_of_accounts"
    CATEGORIZATION_POLICY = "categorization_policy"
    CLIENT_POLICY = "client_policy"
    SOFTWARE_DOCUMENTATION = "software_documentation"


class BookkeepingAuthorityLevel(StrEnum):
    """Optional provenance signal; it never turns guidance into advice."""

    GENERAL_REFERENCE = "general_reference"
    SOFTWARE_VENDOR = "software_vendor"
    INTERNAL_POLICY = "internal_policy"
    CLIENT_APPROVED_POLICY = "client_approved_policy"


class BookkeepingRetrievalMode(StrEnum):
    """Existing retrieval strategies exposed by the bookkeeping bridge."""

    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


_CLIENT_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class BookkeepingKnowledgeMetadata:
    """Validated approval and isolation metadata for one PDF reference."""

    document_type: BookkeepingDocumentType
    title: str
    source_filename: str
    approved_for_bookkeeping: bool
    client_specific: bool
    client_id: str | None = None
    effective_date: date | None = None
    review_date: date | None = None
    authority_level: BookkeepingAuthorityLevel | None = None
    active: bool = True
    domain: str = "bookkeeping"

    def __post_init__(self) -> None:
        if self.domain != "bookkeeping":
            raise BookkeepingMetadataValidationError(
                "domain must be 'bookkeeping'"
            )
        try:
            document_type = BookkeepingDocumentType(self.document_type)
        except (TypeError, ValueError) as error:
            raise BookkeepingMetadataValidationError(
                "document_type is not supported"
            ) from error
        title = _safe_text(self.title, "title", maximum_length=200)
        filename = _safe_filename(self.source_filename)
        if not isinstance(self.approved_for_bookkeeping, bool):
            raise BookkeepingMetadataValidationError(
                "approved_for_bookkeeping must be a boolean"
            )
        if not isinstance(self.client_specific, bool):
            raise BookkeepingMetadataValidationError(
                "client_specific must be a boolean"
            )
        if not isinstance(self.active, bool):
            raise BookkeepingMetadataValidationError(
                "active must be a boolean"
            )

        client_id = self.client_id
        if client_id is not None:
            if not isinstance(client_id, str):
                raise BookkeepingMetadataValidationError(
                    "client_id must be a string or None"
                )
            client_id = client_id.strip().casefold()
            if not _CLIENT_ID_PATTERN.fullmatch(client_id):
                raise BookkeepingMetadataValidationError(
                    "client_id contains unsupported characters"
                )
        if self.client_specific and client_id is None:
            raise BookkeepingMetadataValidationError(
                "client-specific references require client_id"
            )
        if not self.client_specific and client_id is not None:
            raise BookkeepingMetadataValidationError(
                "general references must not contain client_id"
            )

        for field_name, value in (
            ("effective_date", self.effective_date),
            ("review_date", self.review_date),
        ):
            if value is not None and not isinstance(value, date):
                raise BookkeepingMetadataValidationError(
                    f"{field_name} must be a date or None"
                )
        if (
            self.effective_date is not None
            and self.review_date is not None
            and self.review_date < self.effective_date
        ):
            raise BookkeepingMetadataValidationError(
                "review_date cannot precede effective_date"
            )

        authority = self.authority_level
        if authority is not None:
            try:
                authority = BookkeepingAuthorityLevel(authority)
            except (TypeError, ValueError) as error:
                raise BookkeepingMetadataValidationError(
                    "authority_level is not supported"
                ) from error

        object.__setattr__(self, "document_type", document_type)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "source_filename", filename)
        object.__setattr__(self, "client_id", client_id)
        object.__setattr__(self, "authority_level", authority)

    def deactivate(self) -> "BookkeepingKnowledgeMetadata":
        """Return a non-destructive inactive copy."""

        return replace(self, active=False)

    def to_dict(self) -> dict[str, Any]:
        """Return metadata only; document text is intentionally absent."""

        return {
            "domain": self.domain,
            "document_type": self.document_type.value,
            "client_id": self.client_id,
            "client_specific": self.client_specific,
            "title": self.title,
            "source_filename": self.source_filename,
            "effective_date": (
                self.effective_date.isoformat()
                if self.effective_date is not None
                else None
            ),
            "review_date": (
                self.review_date.isoformat()
                if self.review_date is not None
                else None
            ),
            "authority_level": (
                self.authority_level.value
                if self.authority_level is not None
                else None
            ),
            "approved_for_bookkeeping": self.approved_for_bookkeeping,
            "active": self.active,
        }


@dataclass(frozen=True)
class BookkeepingKnowledgeDocument:
    """Safe document status joined to the existing ingestion manifest."""

    manifest_entry: KnowledgeManifestEntry
    classification: BookkeepingKnowledgeMetadata
    source_metadata: DocumentMetadata
    extraction_metadata: Mapping[str, Any]
    page_by_chunk_id: Mapping[str, int | None]

    def __post_init__(self) -> None:
        if self.source_metadata != self.manifest_entry.metadata:
            raise ValueError("source_metadata must match the manifest entry")
        object.__setattr__(
            self,
            "extraction_metadata",
            MappingProxyType(dict(self.extraction_metadata)),
        )
        object.__setattr__(
            self,
            "page_by_chunk_id",
            MappingProxyType(dict(self.page_by_chunk_id)),
        )

    @property
    def document_id(self) -> str:
        return self.manifest_entry.document_id

    @property
    def page_count(self) -> int | None:
        value = self.extraction_metadata.get("page_count")
        return value if isinstance(value, int) and value >= 0 else None

    @property
    def chunk_count(self) -> int:
        return self.manifest_entry.chunk_count

    @property
    def embedding_status(self) -> str:
        return self.manifest_entry.embedding_status


@dataclass(frozen=True)
class BookkeepingCitation:
    """Verified display metadata mapped one-to-one to a retrieved chunk."""

    citation_id: str
    document_id: str
    document_title: str
    source_filename: str
    chunk_id: str
    retrieval_score: float
    page_number: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "citation_id",
            "document_id",
            "document_title",
            "source_filename",
            "chunk_id",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(
                self, name
            ).strip():
                raise ValueError(f"{name} cannot be empty")
        if not math.isfinite(self.retrieval_score):
            raise ValueError("retrieval_score must be finite")
        if self.page_number is not None and (
            isinstance(self.page_number, bool)
            or not isinstance(self.page_number, int)
            or self.page_number <= 0
        ):
            raise ValueError("page_number must be a positive integer or None")


@dataclass(frozen=True)
class BookkeepingRetrievedPassage:
    """Bounded, redacted text and its verified citation."""

    text: str
    citation: BookkeepingCitation
    document_type: BookkeepingDocumentType
    client_id: str | None


@dataclass(frozen=True)
class BookkeepingKnowledgeResult:
    """Structured retrieval outcome, including an explicit no-context state."""

    question: str
    retrieval_mode: BookkeepingRetrievalMode
    passages: tuple[BookkeepingRetrievedPassage, ...]
    relevant_context_found: bool
    sources_conflict: bool = False
    conflict_details: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def citations(self) -> tuple[BookkeepingCitation, ...]:
        return tuple(passage.citation for passage in self.passages)


@dataclass(frozen=True)
class BuiltBookkeepingPrompt:
    """Grounded prompt plus safe construction diagnostics."""

    system_prompt: str
    user_prompt: str
    citation_ids: tuple[str, ...]
    context_characters: int
    calculated_facts: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "calculated_facts",
            MappingProxyType(dict(self.calculated_facts)),
        )


@dataclass(frozen=True)
class GroundedBookkeepingAnswer:
    """Auditable answer result returned by the grounded answer service."""

    answer_text: str
    calculated_facts_used: Mapping[str, str]
    retrieved_citations: tuple[BookkeepingCitation, ...]
    documents_consulted: tuple[str, ...]
    retrieval_mode: BookkeepingRetrievalMode
    provider_name: str
    model_name: str | None
    relevant_context_found: bool
    human_review_required: bool
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    retrieved_passages: tuple[BookkeepingRetrievedPassage, ...] = ()
    sources_conflict: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "calculated_facts_used",
            MappingProxyType(dict(self.calculated_facts_used)),
        )


def _safe_text(value: str, name: str, *, maximum_length: int) -> str:
    if not isinstance(value, str):
        raise BookkeepingMetadataValidationError(f"{name} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise BookkeepingMetadataValidationError(f"{name} cannot be empty")
    if len(normalized) > maximum_length:
        raise BookkeepingMetadataValidationError(f"{name} is too long")
    if _CONTROL_CHARACTER_PATTERN.search(normalized):
        raise BookkeepingMetadataValidationError(
            f"{name} contains control characters"
        )
    return normalized


def _safe_filename(value: str) -> str:
    if not isinstance(value, str):
        raise BookkeepingMetadataValidationError(
            "source_filename must be a string"
        )
    filename = value.strip()
    if not filename:
        raise BookkeepingMetadataValidationError(
            "source_filename cannot be empty"
        )
    if len(filename) > 255:
        raise BookkeepingMetadataValidationError(
            "source_filename is too long"
        )
    if _CONTROL_CHARACTER_PATTERN.search(filename):
        raise BookkeepingMetadataValidationError(
            "source_filename contains control characters"
        )
    if (
        PurePath(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or filename in {".", ".."}
    ):
        raise BookkeepingMetadataValidationError(
            "source_filename cannot contain path components"
        )
    if PurePath(filename).suffix.casefold() != ".pdf":
        raise BookkeepingMetadataValidationError(
            "source_filename must identify a PDF"
        )
    return filename
