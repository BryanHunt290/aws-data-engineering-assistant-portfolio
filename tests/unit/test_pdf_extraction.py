from io import BytesIO
from unittest.mock import patch

import pytest

from knowledge.pdf_extraction import (
    EncryptedPdfError,
    InvalidPdfError,
    NoExtractableTextError,
    PDF_PAGE_SEPARATOR,
    PdfExtractionError,
    PdfTextExtractor,
)
from tests.unit.pdf_fixtures import (
    make_empty_pdf,
    make_encrypted_pdf,
    make_image_only_pdf,
    make_text_pdf,
)


def test_extracts_text_pdf_with_safe_parser_metadata():
    result = PdfTextExtractor().extract_with_metadata(
        make_text_pdf(["Architecture guide"]),
        "pdf",
    )

    assert result.text == "Architecture guide"
    assert result.metadata.page_count == 1
    assert result.metadata.extracted_page_count == 1
    assert result.metadata.pages_with_text == 1
    assert result.metadata.parser_library == "pypdf"
    assert result.metadata.parser_version
    assert result.metadata.encrypted is False
    assert result.metadata.extraction_format == (
        "text/plain; pages-separated-by-form-feed"
    )


def test_preserves_multi_page_order_with_deterministic_separator():
    pdf = make_text_pdf(["First page", "Second page", "Third page"])
    extractor = PdfTextExtractor()

    first = extractor.extract(pdf, "pdf")
    second = extractor.extract(pdf, ".pdf")

    assert first == PDF_PAGE_SEPARATOR.join(
        ["First page", "Second page", "Third page"]
    )
    assert second == first


def test_blank_page_position_and_page_counts_are_preserved():
    result = PdfTextExtractor().extract_with_metadata(
        make_text_pdf(["First page", None, "Third page"]),
        "pdf",
    )

    assert result.text == PDF_PAGE_SEPARATOR.join(
        ["First page", "", "Third page"]
    )
    assert result.metadata.page_count == 3
    assert result.metadata.extracted_page_count == 3
    assert result.metadata.pages_with_text == 2


def test_accepts_binary_stream_and_restores_position():
    stream = BytesIO(make_text_pdf(["Stream input"]))
    stream.seek(7)

    result = PdfTextExtractor().extract_with_metadata(stream, "pdf")

    assert result.text == "Stream input"
    assert stream.tell() == 7


def test_normalizes_all_line_ending_variants():
    assert PdfTextExtractor._normalize_line_endings(
        "one\r\ntwo\rthree\nfour"
    ) == "one\ntwo\nthree\nfour"


def test_pdf_pages_with_no_text_are_rejected():
    with pytest.raises(
        NoExtractableTextError,
        match="no extractable text",
    ):
        PdfTextExtractor().extract(make_text_pdf([None, None]), "pdf")


def test_fully_image_only_pdf_is_rejected_without_ocr():
    with pytest.raises(
        NoExtractableTextError,
        match="image-only PDFs require OCR",
    ):
        PdfTextExtractor().extract(make_image_only_pdf(), "pdf")


def test_empty_pdf_is_rejected():
    with pytest.raises(
        NoExtractableTextError,
        match="contains no pages",
    ):
        PdfTextExtractor().extract(make_empty_pdf(), "pdf")


def test_encrypted_pdf_without_password_is_rejected():
    with pytest.raises(
        EncryptedPdfError,
        match="requires a password",
    ):
        PdfTextExtractor().extract(make_encrypted_pdf(), "pdf")


def test_corrupt_pdf_is_reported_as_invalid():
    with pytest.raises(
        InvalidPdfError,
        match="malformed or corrupted",
    ):
        PdfTextExtractor().extract(b"%PDF-1.7\nnot-a-valid-pdf", "pdf")


def test_unexpected_parser_exception_has_typed_extraction_error():
    with patch(
        "knowledge.pdf_extraction.PdfReader",
        side_effect=RuntimeError("parser defect"),
    ):
        with pytest.raises(
            PdfExtractionError,
            match="text extraction failed",
        ) as raised:
            PdfTextExtractor().extract(b"%PDF", "pdf")

    assert not isinstance(raised.value, InvalidPdfError)


def test_rejects_unsupported_type_and_non_seekable_stream():
    extractor = PdfTextExtractor()

    with pytest.raises(ValueError, match="Unsupported PDF"):
        extractor.extract(b"%PDF", "txt")
    with pytest.raises(TypeError, match="seekable binary stream"):
        extractor.extract(object(), "pdf")
