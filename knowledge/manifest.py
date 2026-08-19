"""JSON manifest persistence with bounded optimistic reconciliation."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from knowledge.indexing_errors import ManifestWriteConflictError
from knowledge.models import KnowledgeManifestEntry
from knowledge.storage import (
    ConditionalKnowledgeStorage,
    ConditionalStorageConflictError,
    KnowledgeKeys,
    KnowledgeStorage,
)


@dataclass(frozen=True)
class IndexingManifestUpdate:
    """Provider-neutral indexing fields persisted after one attempt."""

    embedding_status: str
    index_status: str
    indexed_at: str | None
    embedding_model: str
    embedding_provider: str
    vector_store: str
    vector_dimension: int | None
    indexed_chunk_count: int
    pending_chunk_count: int
    failed_chunk_count: int
    vector_collection: str | None = None


class KnowledgeManifestRepository:
    """Maintain the aggregate manifest without losing concurrent updates."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        storage: KnowledgeStorage,
        *,
        maximum_conflict_retries: int = 3,
    ) -> None:
        if maximum_conflict_retries < 0:
            raise ValueError("maximum_conflict_retries cannot be negative")
        self._storage = storage
        self._maximum_conflict_retries = maximum_conflict_retries

    def upsert(self, entry: KnowledgeManifestEntry) -> None:
        def mutate(manifest: dict[str, Any]) -> dict[str, Any]:
            documents = self._documents(manifest)
            entry_payload = entry.to_dict()
            existing = documents.get(entry.document_id)
            if isinstance(existing, dict):
                existing_metadata = existing.get("metadata")
                if (
                    isinstance(existing_metadata, dict)
                    and existing_metadata.get("checksum")
                    == entry.metadata.checksum
                ):
                    for name in (
                        "embedding_status",
                        "indexed_at",
                        "embedding_model",
                        "embedding_provider",
                        "vector_store",
                        "vector_dimension",
                        "index_status",
                        "indexed_chunk_count",
                        "pending_chunk_count",
                        "failed_chunk_count",
                        "vector_status",
                        "vector_store_provider",
                        "vector_collection",
                    ):
                        if name in existing:
                            entry_payload[name] = existing[name]
            updated = dict(manifest)
            updated["documents"] = {
                **documents,
                entry.document_id: entry_payload,
            }
            return updated

        self._mutate_manifest(mutate, create=True)

    def get(self, document_id: str) -> dict[str, Any] | None:
        manifest = self._storage.get_json(KnowledgeKeys.MANIFEST)
        if manifest is None:
            return None
        entry = self._documents(manifest).get(document_id)
        if entry is not None and not isinstance(entry, dict):
            raise ValueError("Knowledge manifest entry must be an object")
        return entry

    def update_embedding_status(self, document_id: str, status: str) -> None:
        self._update_entry(
            document_id,
            lambda entry: {**entry, "embedding_status": status},
        )

    def update_vector_status(
        self,
        document_id: str,
        status: str,
        *,
        provider: str | None = None,
        collection: str | None = None,
    ) -> None:
        """Record optional vector indexing without replacing ingestion state."""

        self._update_entry(
            document_id,
            lambda entry: {
                **entry,
                "vector_status": status,
                "vector_store_provider": provider,
                "vector_collection": collection,
            },
        )

    def update_indexing_status(
        self,
        document_id: str,
        update: IndexingManifestUpdate,
    ) -> None:
        """Atomically extend one manifest entry with indexing lifecycle data."""

        if min(
            update.indexed_chunk_count,
            update.pending_chunk_count,
            update.failed_chunk_count,
        ) < 0:
            raise ValueError("Indexing manifest counts cannot be negative")
        if update.failed_chunk_count > update.pending_chunk_count:
            raise ValueError("Failed chunks must remain pending")

        self._update_entry(
            document_id,
            lambda entry: {
                **entry,
                "embedding_status": update.embedding_status,
                "indexed_at": update.indexed_at,
                "embedding_model": update.embedding_model,
                "embedding_provider": update.embedding_provider,
                "vector_store": update.vector_store,
                "vector_dimension": update.vector_dimension,
                "index_status": update.index_status,
                "indexed_chunk_count": update.indexed_chunk_count,
                "pending_chunk_count": update.pending_chunk_count,
                "failed_chunk_count": update.failed_chunk_count,
                # Preserve established field names for older readers.
                "vector_status": update.index_status,
                "vector_store_provider": update.vector_store,
                "vector_collection": update.vector_collection,
            },
        )

    def _update_entry(
        self,
        document_id: str,
        updater: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        def mutate(manifest: dict[str, Any]) -> dict[str, Any]:
            documents = self._documents(manifest)
            entry = documents.get(document_id)
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Document '{document_id}' is not present in the manifest"
                )
            updated = dict(manifest)
            updated["documents"] = {
                **documents,
                document_id: updater(dict(entry)),
            }
            return updated

        self._mutate_manifest(mutate)

    def _mutate_manifest(
        self,
        mutator: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        create: bool = False,
    ) -> None:
        if not isinstance(self._storage, ConditionalKnowledgeStorage):
            manifest = self._storage.get_json(KnowledgeKeys.MANIFEST)
            if manifest is None:
                if not create:
                    raise ValueError("Knowledge manifest does not exist")
                manifest = self._new_manifest()
            self._storage.put_json(KnowledgeKeys.MANIFEST, mutator(manifest))
            return

        for attempt in range(self._maximum_conflict_retries + 1):
            versioned = self._storage.get_json_versioned(KnowledgeKeys.MANIFEST)
            manifest = versioned.payload
            if manifest is None:
                if not create:
                    raise ValueError("Knowledge manifest does not exist")
                manifest = self._new_manifest()
            updated = mutator(manifest)
            try:
                self._storage.put_json_if_version(
                    KnowledgeKeys.MANIFEST,
                    updated,
                    versioned.version,
                )
                return
            except ConditionalStorageConflictError as error:
                if attempt == self._maximum_conflict_retries:
                    raise ManifestWriteConflictError(
                        "Manifest update exceeded the conflict retry limit"
                    ) from error

    @classmethod
    def _new_manifest(cls) -> dict[str, Any]:
        return {"schema_version": cls.SCHEMA_VERSION, "documents": {}}

    @staticmethod
    def _documents(manifest: dict[str, Any]) -> dict[str, Any]:
        documents = manifest.get("documents")
        if not isinstance(documents, dict):
            raise ValueError("Knowledge manifest documents must be an object")
        return documents
