"""Safe, provider-neutral inspection and redrive planning."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from knowledge.indexing_errors import RedriveSafetyError
from knowledge.storage import (
    ConditionalKnowledgeStorage,
    ConditionalStorageConflictError,
    KnowledgeKeys,
    KnowledgeStorage,
)


PERMANENT_FAILURE_TYPES = frozenset(
    {
        "IndexingBatchLimitError",
        "MissingClientFilterError",
        "VectorDimensionMismatchError",
        "ValueError",
    }
)


@dataclass(frozen=True)
class RedriveFilters:
    """Required scope plus optional identifier/status constraints."""

    client_id: str
    environment: str
    namespace: str | None = None
    domain: str | None = None
    document_id: str | None = None
    status: str | None = None


@dataclass(frozen=True)
class RedriveCandidate:
    document_id: str
    descriptor_key: str
    raw_key: str
    indexed_count: int
    pending_count: int
    failed_count: int
    retryable_count: int
    permanent_count: int


@dataclass(frozen=True)
class RedriveReport:
    dry_run: bool
    candidates: tuple[RedriveCandidate, ...]
    descriptors_updated: int
    documents_dispatched: int


class IndexingRedriveService:
    """Inspect descriptors and explicitly reset/dispatch retryable work."""

    def __init__(self, storage: KnowledgeStorage) -> None:
        self._storage = storage

    def inspect(self, filters: RedriveFilters) -> tuple[RedriveCandidate, ...]:
        self._validate_filters(filters)
        manifest = self._storage.get_json(KnowledgeKeys.MANIFEST)
        documents = manifest.get("documents") if manifest is not None else {}
        if not isinstance(documents, dict):
            raise RedriveSafetyError("Knowledge manifest is malformed")
        candidates: list[RedriveCandidate] = []
        for document_id, raw_entry in sorted(documents.items()):
            if not isinstance(raw_entry, dict):
                continue
            if filters.document_id and document_id != filters.document_id:
                continue
            descriptor_key = raw_entry.get("embedding_key")
            raw_key = raw_entry.get("raw_key")
            if not isinstance(descriptor_key, str) or not isinstance(raw_key, str):
                continue
            descriptor = self._storage.get_json(descriptor_key)
            if descriptor is None:
                continue
            self._assert_scope(descriptor, filters)
            if filters.namespace and descriptor.get("namespace") != filters.namespace:
                continue
            if filters.domain and descriptor.get("domain") != filters.domain:
                continue
            if filters.status and descriptor.get("index_status") != filters.status:
                continue
            states = descriptor.get("chunks")
            if not isinstance(states, list):
                raise RedriveSafetyError("Indexing descriptor is malformed")
            indexed = sum(
                isinstance(state, dict) and state.get("status") == "indexed"
                for state in states
            )
            pending = len(states) - indexed
            failure_types = [
                state.get("last_error_type")
                for state in states
                if isinstance(state, dict)
                and state.get("status") != "indexed"
                and state.get("last_error_type")
            ]
            permanent = sum(value in PERMANENT_FAILURE_TYPES for value in failure_types)
            retryable = len(failure_types) - permanent
            if pending:
                candidates.append(
                    RedriveCandidate(
                        document_id=document_id,
                        descriptor_key=descriptor_key,
                        raw_key=raw_key,
                        indexed_count=indexed,
                        pending_count=pending,
                        failed_count=len(failure_types),
                        retryable_count=retryable,
                        permanent_count=permanent,
                    )
                )
        return tuple(candidates)

    def redrive(
        self,
        filters: RedriveFilters,
        *,
        apply: bool = False,
        reset_retryable: bool = False,
        dispatcher: Callable[[RedriveCandidate], None] | None = None,
    ) -> RedriveReport:
        candidates = self.inspect(filters)
        if not apply:
            return RedriveReport(True, candidates, 0, 0)
        if dispatcher is None:
            raise RedriveSafetyError("Apply mode requires an explicit dispatcher")
        updated = dispatched = 0
        for candidate in candidates:
            if candidate.retryable_count == 0:
                continue
            if reset_retryable:
                if not self._reset_retryable_descriptor(candidate):
                    continue
                updated += 1
            dispatcher(candidate)
            dispatched += 1
        return RedriveReport(False, candidates, updated, dispatched)

    def _reset_retryable_descriptor(
        self, candidate: RedriveCandidate
    ) -> bool:
        for attempt in range(4):
            if isinstance(self._storage, ConditionalKnowledgeStorage):
                versioned = self._storage.get_json_versioned(
                    candidate.descriptor_key
                )
                descriptor = versioned.payload
                version = versioned.version
            else:
                descriptor = self._storage.get_json(candidate.descriptor_key)
                version = None
            if descriptor is None:
                return False
            states = descriptor.get("chunks")
            if not isinstance(states, list):
                raise RedriveSafetyError("Indexing descriptor is malformed")
            changed = False
            indexed_count = 0
            for state in states:
                if not isinstance(state, dict):
                    continue
                if state.get("status") == "indexed":
                    indexed_count += 1
                    continue
                failure_type = state.get("last_error_type")
                if failure_type and failure_type not in PERMANENT_FAILURE_TYPES:
                    state["last_error_type"] = None
                    state["last_failure_stage"] = None
                    state["status"] = "pending"
                    changed = True
            if not changed:
                return False
            descriptor["index_status"] = (
                "partial" if indexed_count else "pending"
            )
            try:
                if isinstance(self._storage, ConditionalKnowledgeStorage):
                    self._storage.put_json_if_version(
                        candidate.descriptor_key, descriptor, version
                    )
                else:
                    self._storage.put_json(candidate.descriptor_key, descriptor)
                return True
            except ConditionalStorageConflictError as error:
                if attempt == 3:
                    raise RedriveSafetyError(
                        "Redrive descriptor conflict retry limit exceeded"
                    ) from error
        return False  # pragma: no cover

    @staticmethod
    def _validate_filters(filters: RedriveFilters) -> None:
        if not filters.client_id.strip() or not filters.environment.strip():
            raise RedriveSafetyError("Client and environment scope are required")

    @staticmethod
    def _assert_scope(
        descriptor: dict[str, Any], filters: RedriveFilters
    ) -> None:
        if descriptor.get("client_id") != filters.client_id:
            raise RedriveSafetyError("Refusing cross-client redrive operation")
        if descriptor.get("environment") != filters.environment:
            raise RedriveSafetyError("Refusing cross-environment redrive operation")
