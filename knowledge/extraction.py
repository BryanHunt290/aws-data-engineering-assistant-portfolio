"""Text extraction interfaces and registry for supported document formats."""

from dataclasses import dataclass, field
from typing import Any
from typing import Protocol, runtime_checkable


TEXT_DOCUMENT_TYPES = frozenset(
    {
        "html",
        "json",
        "md",
        "markdown",
        "py",
        "txt",
    }
)
EXTRACTABLE_DOCUMENT_TYPES = TEXT_DOCUMENT_TYPES | frozenset({"pdf"})


@dataclass(frozen=True)
class DocumentExtraction:
    """Chunkable text plus optional non-content extraction metadata."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DocumentTextExtractor(Protocol):
    """Interface for converting a supported document into chunkable text."""

    def supports(self, file_type: str) -> bool:
        """Return whether this extractor handles the file type."""

    def extract(self, content: bytes, file_type: str) -> str:
        """Extract text without losing the original document."""


class Utf8TextExtractor:
    """Baseline extractor for UTF-8 text-based document types."""

    def supports(self, file_type: str) -> bool:
        return file_type in TEXT_DOCUMENT_TYPES

    def extract(self, content: bytes, file_type: str) -> str:
        if not self.supports(file_type):
            raise ValueError(f"Unsupported text extraction type '{file_type}'")
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"{file_type} document is not valid UTF-8"
            ) from error


class ExtractorRegistry:
    """Select the first registered extractor that supports a document type."""

    def __init__(
        self,
        extractors: tuple[DocumentTextExtractor, ...] | None = None,
    ) -> None:
        if extractors is None:
            from knowledge.pdf_extraction import PdfTextExtractor

            extractors = (Utf8TextExtractor(), PdfTextExtractor())
        self._extractors = extractors

    def extract(self, content: bytes, file_type: str) -> str | None:
        """Extract text while preserving the original public API."""

        result = self.extract_with_metadata(content, file_type)
        return result.text if result is not None else None

    def extract_with_metadata(
        self,
        content: bytes,
        file_type: str,
    ) -> DocumentExtraction | None:
        """Extract text and any safe, format-specific metadata."""

        for extractor in self._extractors:
            if extractor.supports(file_type):
                detailed_extract = getattr(
                    extractor,
                    "extract_with_metadata",
                    None,
                )
                if callable(detailed_extract):
                    result = detailed_extract(content, file_type)
                    return DocumentExtraction(
                        text=result.text,
                        metadata=result.metadata.to_dict(),
                    )
                return DocumentExtraction(
                    text=extractor.extract(content, file_type)
                )
        return None


# TODO(knowledge-extraction): Add a DOCX extractor behind
# DocumentTextExtractor without changing the ingestion pipeline.
