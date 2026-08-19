"""Minimal local asset bundling for the document-ingestion Lambda."""

from importlib.util import find_spec
from pathlib import Path
import shutil
from typing import Any

import jsii
from aws_cdk import ILocalBundling


@jsii.implements(ILocalBundling)
class DocumentIngestionAssetBundler:
    """Bundle application modules and pinned runtime dependencies."""

    _RUNTIME_PACKAGES = (
        "pypdf",
        "requests",
        "certifi",
        "charset_normalizer",
        "idna",
        "urllib3",
    )

    def __init__(self, source_root: Path | None = None) -> None:
        self._source_root = (
            source_root
            or Path(__file__).resolve().parents[1]
        )

    def try_bundle(self, output_dir: str, options: Any) -> bool:
        """Create a small Lambda asset without tests or local environments."""

        del options
        destination = Path(output_dir)
        self._copy_source_directory("knowledge", destination)
        self._copy_source_directory(
            "lambda/document_ingestion",
            destination,
        )

        for package_name in self._RUNTIME_PACKAGES:
            self._copy_installed_package(package_name, destination)
        return True

    def _copy_installed_package(
        self,
        package_name: str,
        destination: Path,
    ) -> None:
        package_spec = find_spec(package_name)
        package_locations = (
            package_spec.submodule_search_locations
            if package_spec is not None
            else None
        )
        if not package_locations:
            raise RuntimeError(
                f"{package_name} is required to bundle the "
                "document-ingestion Lambda; install requirements.txt "
                "with constraints.txt"
            )
        package_source = Path(next(iter(package_locations)))
        shutil.copytree(
            package_source,
            destination / package_name,
            dirs_exist_ok=True,
            ignore=self._ignored_files,
        )

    def _copy_source_directory(
        self,
        relative_path: str,
        destination: Path,
    ) -> None:
        source = self._source_root / relative_path
        if not source.is_dir():
            raise RuntimeError(
                f"Lambda asset source directory is missing: {relative_path}"
            )
        shutil.copytree(
            source,
            destination / relative_path,
            dirs_exist_ok=True,
            ignore=self._ignored_files,
        )

    @staticmethod
    def _ignored_files(
        directory: str,
        names: list[str],
    ) -> set[str]:
        del directory
        return {
            name
            for name in names
            if name == "__pycache__"
            or name in {".pytest_cache", ".mypy_cache", ".ruff_cache"}
            or Path(name).suffix in {".pyc", ".pyo", ".pyd"}
        }
