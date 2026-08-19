"""Local, page-ordered text extraction for PDF documents."""

from contextlib import suppress
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any, BinaryIO

import pypdf
from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PdfReadError


PDF_DOCUMENT_TYPES = frozenset({"pdf"})
PDF_PAGE_SEPARATOR = "\n\n\f\n\n"


class PdfExtractionError(RuntimeError):
    """Base error for a PDF that cannot produce ingestion text."""


class InvalidPdfError(PdfExtractionError):
    """Raised when input is not a structurally readable PDF."""


class EncryptedPdfError(PdfExtractionError):
    """Raised when a PDF requires a password that ingestion does not have."""


class NoExtractableTextError(PdfExtractionError):
    """Raised when a readable PDF contains no meaningful text."""


@dataclass(frozen=True)
class PdfExtractionMetadata:
    """Non-content metadata recorded for a successfully extracted PDF."""

    page_count: int
    extracted_page_count: int
    pages_with_text: int
    parser_library: str
    parser_version: str
    encrypted: bool
    extraction_format: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PdfExtractionResult:
    """Extracted PDF text and safe parser metadata."""

    text: str
    metadata: PdfExtractionMetadata


class PdfTextExtractor:
    """Extract text from a PDF locally without OCR or embedded execution."""

    def supports(self, file_type: str) -> bool:
        return file_type.lower().lstrip(".") in PDF_DOCUMENT_TYPES

    def extract(self, content: bytes | BinaryIO, file_type: str) -> str:
        """Return text for compatibility with the existing extractor API."""

        return self.extract_with_metadata(content, file_type).text

    def extract_with_metadata(
        self,
        content: bytes | BinaryIO,
        file_type: str,
    ) -> PdfExtractionResult:
        """Extract pages in source order and return non-content metadata."""

        if not self.supports(file_type):
            raise ValueError(f"Unsupported PDF extraction type '{file_type}'")

        stream, original_position = self._prepare_stream(content)
        try:
            return self._extract_stream(stream)
        finally:
            if original_position is not None:
                with suppress(Exception):
                    stream.seek(original_position)

    def _extract_stream(self, stream: BinaryIO) -> PdfExtractionResult:
        try:
            reader = PdfReader(stream, strict=True)
            if reader.is_encrypted:
                raise EncryptedPdfError(
                    "Encrypted PDF requires a password; password-based "
                    "ingestion is not configured"
                )

            page_count = len(reader.pages)
            if page_count == 0:
                raise NoExtractableTextError(
                    "PDF contains no pages and no extractable text"
                )

            page_texts: list[str] = []
            extracted_page_count = 0
            pages_with_text = 0
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text is None:
                    normalized_text = ""
                else:
                    extracted_page_count += 1
                    normalized_text = self._normalize_line_endings(page_text)
                if normalized_text.strip():
                    pages_with_text += 1
                page_texts.append(normalized_text)

            text = PDF_PAGE_SEPARATOR.join(page_texts)
            if not text.strip():
                raise NoExtractableTextError(
                    "PDF pages contain no extractable text; scanned or "
                    "image-only PDFs require OCR, which is not enabled"
                )

            return PdfExtractionResult(
                text=text,
                metadata=PdfExtractionMetadata(
                    page_count=page_count,
                    extracted_page_count=extracted_page_count,
                    pages_with_text=pages_with_text,
                    parser_library="pypdf",
                    parser_version=pypdf.__version__,
                    encrypted=False,
                    extraction_format=(
                        "text/plain; pages-separated-by-form-feed"
                    ),
                ),
            )
        except (EncryptedPdfError, NoExtractableTextError):
            raise
        except FileNotDecryptedError as error:
            raise EncryptedPdfError(
                "Encrypted PDF requires a password; password-based "
                "ingestion is not configured"
            ) from error
        except PdfReadError as error:
            raise InvalidPdfError("PDF is malformed or corrupted") from error
        except Exception as error:
            raise PdfExtractionError("PDF text extraction failed") from error

    @staticmethod
    def _prepare_stream(
        content: bytes | BinaryIO,
    ) -> tuple[BinaryIO, int | None]:
        if isinstance(content, bytes):
            return BytesIO(content), None
        if not hasattr(content, "read") or not hasattr(content, "seek"):
            raise TypeError(
                "PDF content must be bytes or a seekable binary stream"
            )
        try:
            original_position = content.tell()
            content.seek(0)
        except (AttributeError, OSError, TypeError, ValueError) as error:
            raise TypeError(
                "PDF content must be bytes or a seekable binary stream"
            ) from error
        return content, original_position

    @staticmethod
    def _normalize_line_endings(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")
