"""Fail-closed classification for uploaded knowledge objects.

The classifier is deterministic and provider-neutral. It inspects only the
declared MIME type, normalized filename extension, and supplied object bytes.
It never performs OCR, transcription, captioning, embedding, or network I/O.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import json
from pathlib import PurePath
import re
from typing import Collection, Mapping


class ObjectClassification(StrEnum):
    """The only categories accepted by the ingestion boundary."""

    INDEXABLE_TEXT_DOCUMENT = "indexable_text_document"
    MEDIA_OBJECT = "media_object"
    UNSUPPORTED_BINARY = "unsupported_binary"
    REJECTED_OR_SUSPICIOUS = "rejected_or_suspicious"


class MediaType(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    OTHER = "other"


class NonIndexableObjectError(ValueError):
    """Raised when a storage-only object reaches a knowledge-indexing stage."""


INDEXABLE_TEXT_EXTENSIONS = frozenset(
    {"html", "json", "markdown", "md", "pdf", "py", "txt"}
)
IMAGE_EXTENSIONS = frozenset(
    {"jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff", "svg", "heic"}
)
VIDEO_EXTENSIONS = frozenset(
    {"mp4", "mov", "avi", "mkv", "webm", "mpeg", "mpg", "m4v"}
)
AUDIO_EXTENSIONS = frozenset(
    {"mp3", "wav", "m4a", "aac", "flac", "ogg"}
)
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

_GENERIC_MIME_TYPES = frozenset(
    {"", "application/octet-stream", "binary/octet-stream"}
)
_EXPECTED_MIME_TYPES: dict[str, frozenset[str]] = {
    "txt": frozenset({"text/plain"}),
    "md": frozenset({"text/markdown", "text/plain", "text/x-markdown"}),
    "markdown": frozenset(
        {"text/markdown", "text/plain", "text/x-markdown"}
    ),
    "html": frozenset({"text/html", "application/xhtml+xml"}),
    "json": frozenset({"application/json", "text/json", "text/plain"}),
    "py": frozenset({"text/plain", "text/x-python", "application/x-python-code"}),
    "pdf": frozenset({"application/pdf"}),
    "jpg": frozenset({"image/jpeg", "image/jpg"}),
    "jpeg": frozenset({"image/jpeg", "image/jpg"}),
    "png": frozenset({"image/png"}),
    "gif": frozenset({"image/gif"}),
    "webp": frozenset({"image/webp"}),
    "bmp": frozenset({"image/bmp", "image/x-ms-bmp"}),
    "tiff": frozenset({"image/tiff"}),
    "svg": frozenset({"image/svg+xml"}),
    "heic": frozenset({"image/heic", "image/heif"}),
    "mp4": frozenset({"video/mp4"}),
    "m4v": frozenset({"video/mp4", "video/x-m4v"}),
    "mov": frozenset({"video/quicktime"}),
    "avi": frozenset({"video/x-msvideo", "video/avi"}),
    "mkv": frozenset({"video/x-matroska"}),
    "webm": frozenset({"video/webm"}),
    "mpeg": frozenset({"video/mpeg"}),
    "mpg": frozenset({"video/mpeg"}),
    "mp3": frozenset({"audio/mpeg", "audio/mp3"}),
    "wav": frozenset({"audio/wav", "audio/x-wav"}),
    "m4a": frozenset({"audio/mp4", "audio/x-m4a"}),
    "aac": frozenset({"audio/aac", "audio/x-aac"}),
    "flac": frozenset({"audio/flac", "audio/x-flac"}),
    "ogg": frozenset({"audio/ogg", "application/ogg"}),
}
_MIME_EXTENSIONS: dict[str, frozenset[str]] = {
    mime_type: frozenset(
        extension
        for extension, values in _EXPECTED_MIME_TYPES.items()
        if mime_type in values
    )
    for mime_type in {
        value for values in _EXPECTED_MIME_TYPES.values() for value in values
    }
}


@dataclass(frozen=True)
class ObjectInspection:
    """Safe classification facts propagated through guarded stages."""

    object_classification: ObjectClassification
    detected_mime_type: str
    declared_mime_type: str | None
    file_extension: str
    media_type: MediaType | None
    storage_only: bool
    indexable: bool
    quarantine_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["object_classification"] = self.object_classification.value
        payload["media_type"] = (
            self.media_type.value if self.media_type is not None else None
        )
        return payload


def classify_uploaded_object(
    *,
    filename: str,
    declared_mime_type: str | None,
    content: bytes,
    supported_document_types: Collection[str] = INDEXABLE_TEXT_EXTENSIONS,
) -> ObjectInspection:
    """Classify bytes using extension, MIME metadata, and file signatures."""

    if not isinstance(content, bytes):
        raise TypeError("content must be bytes")
    if not filename.strip():
        raise ValueError("filename cannot be empty")
    if PurePath(filename).name != filename or "\\" in filename:
        raise ValueError("filename cannot contain path components")
    extension = PurePath(filename).suffix.lower().lstrip(".")
    declared = _normalize_mime_type(declared_mime_type)
    detected_mime, detected_extensions, dangerous = _detect_content(content)
    approved = INDEXABLE_TEXT_EXTENSIONS.intersection(
        value.strip().lower().lstrip(".")
        for value in supported_document_types
    )

    if dangerous:
        return _rejected(
            detected_mime,
            declared,
            extension,
            "dangerous_executable_signature",
        )

    if declared not in _GENERIC_MIME_TYPES:
        expected = _EXPECTED_MIME_TYPES.get(extension, frozenset())
        if expected and declared not in expected:
            return _rejected(
                detected_mime,
                declared,
                extension,
                "declared_mime_type_mismatch",
            )
        declared_extensions = _MIME_EXTENSIONS.get(declared, frozenset())
        if declared_extensions and extension not in declared_extensions:
            return _rejected(
                detected_mime,
                declared,
                extension,
                "declared_mime_type_mismatch",
            )

    if extension in approved:
        if extension not in detected_extensions:
            return _rejected(
                detected_mime,
                declared,
                extension,
                "extension_signature_mismatch",
            )
        if (
            declared not in _GENERIC_MIME_TYPES
            and declared not in _EXPECTED_MIME_TYPES[extension]
        ):
            return _rejected(
                detected_mime,
                declared,
                extension,
                "declared_mime_type_mismatch",
            )
        return ObjectInspection(
            object_classification=(
                ObjectClassification.INDEXABLE_TEXT_DOCUMENT
            ),
            detected_mime_type=detected_mime,
            declared_mime_type=declared or None,
            file_extension=extension,
            media_type=None,
            storage_only=False,
            indexable=True,
        )

    if extension in MEDIA_EXTENSIONS:
        if extension not in detected_extensions:
            return _rejected(
                detected_mime,
                declared,
                extension,
                "extension_signature_mismatch",
            )
        return ObjectInspection(
            object_classification=ObjectClassification.MEDIA_OBJECT,
            detected_mime_type=detected_mime,
            declared_mime_type=declared or None,
            file_extension=extension,
            media_type=_media_type(extension),
            storage_only=True,
            indexable=False,
        )

    if detected_extensions.intersection(MEDIA_EXTENSIONS):
        return _rejected(
            detected_mime,
            declared,
            extension,
            "extension_signature_mismatch",
        )
    return ObjectInspection(
        object_classification=ObjectClassification.UNSUPPORTED_BINARY,
        detected_mime_type=detected_mime,
        declared_mime_type=declared or None,
        file_extension=extension,
        media_type=MediaType.OTHER,
        storage_only=True,
        indexable=False,
    )


def require_indexable_object(
    classification: ObjectInspection | ObjectClassification | str,
    *,
    stage: str,
) -> None:
    """Fail closed when a non-document reaches a knowledge-indexing stage."""

    value = (
        classification.object_classification
        if isinstance(classification, ObjectInspection)
        else classification
    )
    try:
        parsed = ObjectClassification(value)
    except (TypeError, ValueError) as error:
        raise NonIndexableObjectError(
            f"{stage} requires a recognized object classification"
        ) from error
    if parsed is not ObjectClassification.INDEXABLE_TEXT_DOCUMENT:
        raise NonIndexableObjectError(
            f"{stage} blocked object classification '{parsed.value}'"
        )


def require_indexable_metadata(
    metadata: Mapping[str, object],
    *,
    stage: str,
) -> None:
    """Validate redundant indexability fields at a persistence boundary."""

    require_indexable_object(
        metadata.get("object_classification", ""),
        stage=stage,
    )
    if metadata.get("indexable") is not True:
        raise NonIndexableObjectError(f"{stage} requires indexable=true")
    if metadata.get("storage_only") is not False:
        raise NonIndexableObjectError(f"{stage} requires storage_only=false")


def _normalize_mime_type(value: str | None) -> str:
    if not isinstance(value, str):
        return ""
    return value.split(";", 1)[0].strip().lower()


def _rejected(
    detected_mime_type: str,
    declared_mime_type: str,
    extension: str,
    reason: str,
) -> ObjectInspection:
    media_type = _detected_media_type(detected_mime_type)
    if media_type is None and extension in MEDIA_EXTENSIONS:
        media_type = _media_type(extension)
    return ObjectInspection(
        object_classification=ObjectClassification.REJECTED_OR_SUSPICIOUS,
        detected_mime_type=detected_mime_type,
        declared_mime_type=declared_mime_type or None,
        file_extension=extension,
        media_type=media_type,
        storage_only=True,
        indexable=False,
        quarantine_reason=reason,
    )


def _media_type(extension: str) -> MediaType:
    if extension in IMAGE_EXTENSIONS:
        return MediaType.IMAGE
    if extension in VIDEO_EXTENSIONS:
        return MediaType.VIDEO
    if extension in AUDIO_EXTENSIONS:
        return MediaType.AUDIO
    return MediaType.OTHER


def _detected_media_type(mime_type: str) -> MediaType | None:
    if mime_type.startswith("image/"):
        return MediaType.IMAGE
    if mime_type.startswith("video/"):
        return MediaType.VIDEO
    if mime_type.startswith("audio/"):
        return MediaType.AUDIO
    return None


def _detect_content(content: bytes) -> tuple[str, frozenset[str], bool]:
    head = content[:8192]
    if head.startswith(b"%PDF-"):
        return "application/pdf", frozenset({"pdf"}), False
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", frozenset({"jpg", "jpeg"}), False
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", frozenset({"png"}), False
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif", frozenset({"gif"}), False
    if head.startswith(b"BM"):
        return "image/bmp", frozenset({"bmp"}), False
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff", frozenset({"tiff"}), False
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "image/webp", frozenset({"webp"}), False
    if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        return "audio/wav", frozenset({"wav"}), False
    if head.startswith(b"RIFF") and head[8:12] == b"AVI ":
        return "video/x-msvideo", frozenset({"avi"}), False
    if head.startswith(b"fLaC"):
        return "audio/flac", frozenset({"flac"}), False
    if head.startswith(b"OggS"):
        return "audio/ogg", frozenset({"ogg"}), False
    if head.startswith(b"ID3") or (
        len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE6 == 0xE2
    ):
        return "audio/mpeg", frozenset({"mp3"}), False
    if head.startswith(b"ADIF") or (
        len(head) >= 2 and head[0] == 0xFF and head[1] & 0xF6 == 0xF0
    ):
        return "audio/aac", frozenset({"aac"}), False
    if head.startswith((b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3")):
        return "video/mpeg", frozenset({"mpeg", "mpg"}), False
    if head.startswith(b"\x1aE\xdf\xa3"):
        lowered = head.lower()
        if b"webm" in lowered:
            return "video/webm", frozenset({"webm"}), False
        return "video/x-matroska", frozenset({"mkv"}), False
    if len(head) >= 12 and head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"}:
            return "image/heic", frozenset({"heic"}), False
        if brand in {b"M4A ", b"M4B ", b"M4P "}:
            return "audio/mp4", frozenset({"m4a"}), False
        if brand == b"qt  ":
            return "video/quicktime", frozenset({"mov"}), False
        return "video/mp4", frozenset({"mp4", "m4v"}), False
    if head.startswith(b"MZ") or head.startswith(b"\x7fELF"):
        return "application/x-executable", frozenset(), True
    if head.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        return "application/zip", frozenset(), False

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return "application/octet-stream", frozenset(), False
    if "\x00" in text or _has_unsafe_control_density(text):
        return "application/octet-stream", frozenset(), False
    stripped = text.lstrip("\ufeff\t\r\n ")
    if re.match(r"(?is)(?:<\?xml[^>]*>\s*)?<svg(?:\s|>)", stripped):
        return "image/svg+xml", frozenset({"svg"}), False
    if re.match(
        r"(?is)(?:<!doctype\s+html|<(?:html|head|body|main|section|article|"
        r"div|p|h[1-6]|table|ul|ol)(?:\s|>))",
        stripped,
    ):
        return "text/html", frozenset({"html"}), False
    json_valid = True
    try:
        json.loads(text)
    except json.JSONDecodeError:
        json_valid = False
    if json_valid:
        return "application/json", frozenset({"json"}), False
    return (
        "text/plain",
        frozenset({"txt", "md", "markdown", "py"}),
        False,
    )


def _has_unsafe_control_density(text: str) -> bool:
    if not text:
        return False
    controls = sum(
        ord(character) < 32 and character not in "\t\n\r\f"
        for character in text
    )
    return controls / len(text) > 0.01
