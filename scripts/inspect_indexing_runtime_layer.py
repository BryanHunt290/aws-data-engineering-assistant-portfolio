"""Inspect a Lambda layer ZIP without importing or executing its contents."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import BadZipFile, ZipFile


_NATIVE_SUFFIXES = (".so", ".pyd", ".dll", ".dylib")


@dataclass(frozen=True)
class LayerInspectionReport:
    valid: bool
    archive_name: str
    archive_size_bytes: int
    sha256: str
    file_count: int
    qdrant_client_version: str | None
    python_root_present: bool
    native_file_count: int
    native_suffixes: tuple[str, ...]
    errors: tuple[str, ...]


def inspect_layer_archive(
    archive_path: Path,
    *,
    expected_qdrant_version: str,
) -> LayerInspectionReport:
    content = archive_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    errors: list[str] = []
    version: str | None = None
    native_suffixes: set[str] = set()
    try:
        with ZipFile(archive_path) as archive:
            names = tuple(archive.namelist())
            python_root = bool(names) and all(
                PurePosixPath(name).parts[:1] == ("python",)
                for name in names
            )
            if not python_root:
                errors.append("archive_must_contain_only_python_root")
            metadata_names = [
                name
                for name in names
                if re.fullmatch(
                    r"python/qdrant_client-[^/]+\.dist-info/METADATA",
                    name,
                    re.IGNORECASE,
                )
            ]
            if len(metadata_names) != 1:
                errors.append("qdrant_client_metadata_missing_or_ambiguous")
            else:
                metadata = archive.read(metadata_names[0]).decode(
                    "utf-8", errors="replace"
                )
                match = re.search(r"^Version:\s*(\S+)\s*$", metadata, re.MULTILINE)
                version = match.group(1) if match else None
                if version != expected_qdrant_version:
                    errors.append("qdrant_client_version_mismatch")
            for name in names:
                suffix = PurePosixPath(name).suffix.casefold()
                if suffix in _NATIVE_SUFFIXES:
                    native_suffixes.add(suffix)
    except BadZipFile:
        names = ()
        python_root = False
        errors.append("archive_is_not_a_zip")
    return LayerInspectionReport(
        valid=not errors,
        archive_name=archive_path.name,
        archive_size_bytes=len(content),
        sha256=digest,
        file_count=len(names),
        qdrant_client_version=version,
        python_root_present=python_root,
        native_file_count=sum(
            PurePosixPath(name).suffix.casefold() in _NATIVE_SUFFIXES
            for name in names
        ),
        native_suffixes=tuple(sorted(native_suffixes)),
        errors=tuple(errors),
    )


def expected_qdrant_version(requirements_path: Path) -> str:
    match = re.search(
        r"^qdrant-client==([^\s#]+)",
        requirements_path.read_text(encoding="utf-8"),
        re.MULTILINE | re.IGNORECASE,
    )
    if match is None:
        raise ValueError("Pinned qdrant-client requirement was not found")
    return match.group(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument(
        "--requirements",
        type=Path,
        default=Path("lambda/indexing_runtime_requirements.txt"),
    )
    args = parser.parse_args()
    try:
        report = inspect_layer_archive(
            args.archive,
            expected_qdrant_version=expected_qdrant_version(
                args.requirements
            ),
        )
    except (OSError, ValueError):
        print(json.dumps({"valid": False, "errors": ["inspection_input_invalid"]}))
        return 2
    print(json.dumps(asdict(report), sort_keys=True))
    return 0 if report.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
