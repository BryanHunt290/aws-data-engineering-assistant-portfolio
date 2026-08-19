"""Offline preflight, phase planning, and evidence verification."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
import re
import sys
from typing import Any, Protocol

from config.clients import ClientConfig, resolve_client_config
from knowledge.indexing_configuration import validate_qdrant_endpoint
from knowledge.indexing_errors import IntegrationValidationError


_BEDROCK_ARN = re.compile(
    r"arn:(?:aws|aws-us-gov):bedrock:[a-z0-9-]+:(?:\d{12})?:"
    r"(?:foundation-model|inference-profile|application-inference-profile)/"
    r"[A-Za-z0-9._:/+-]+"
)
_SECRET_ARN = re.compile(
    r"arn:(?:aws|aws-us-gov):secretsmanager:[a-z0-9-]+:\d{12}:"
    r"secret:[A-Za-z0-9/_+=.@-]+"
)
_LAYER_ARN = re.compile(
    r"arn:(?:aws|aws-us-gov):lambda:[a-z0-9-]+:\d{12}:"
    r"layer:[A-Za-z0-9_-]+:[1-9]\d*"
)
_KMS_ARN = re.compile(
    r"arn:(?:aws|aws-us-gov):kms:[a-z0-9-]+:\d{12}:"
    r"key/[A-Za-z0-9-]+"
)
_COLLECTION = re.compile(r"[a-z0-9](?:[a-z0-9_]{0,118}[a-z0-9])?")
_SAFE_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")
_PLAINTEXT_SECRET_KEYS = frozenset(
    {
        "appqdrantapikey",
        "indexingqdrantapikey",
        "knowledgeqdrantapikey",
        "qdrantapikey",
    }
)


@dataclass(frozen=True)
class IntegrationPreflightReport:
    """Safe configuration-only result containing no external values."""

    ready: bool
    enabled: bool
    client_id: str | None
    environment: str | None
    checks: tuple[str, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "enabled": self.enabled,
            "client_id": self.client_id,
            "environment": self.environment,
            "check_count": len(self.checks),
            "checks": list(self.checks),
            "error_count": len(self.errors),
            "errors": list(self.errors),
            "network_calls": 0,
        }


class IntegrationPreflightValidator:
    """Validate existing CDK/indexing context without provider calls."""

    def __init__(
        self,
        *,
        python_version: Callable[[], tuple[int, int]] | None = None,
        config_resolver: Callable[
            [Mapping[str, object]], ClientConfig
        ] = resolve_client_config,
    ) -> None:
        self._python_version = python_version or (
            lambda: (sys.version_info.major, sys.version_info.minor)
        )
        self._config_resolver = config_resolver

    def validate(
        self, context: Mapping[str, object]
    ) -> IntegrationPreflightReport:
        errors: list[str] = []
        checks: list[str] = []
        client: ClientConfig | None = None

        if self._python_version() == (3, 12):
            checks.append("python_3_12")
        else:
            errors.append("python_version_unsupported")

        if self._contains_plaintext_secret(context):
            errors.append("plaintext_api_key_forbidden")
        else:
            checks.append("plaintext_api_key_absent")

        for key in ("client", "clientId", "environment"):
            if not isinstance(context.get(key), str) or not str(
                context[key]
            ).strip():
                errors.append(f"explicit_{key}_required")

        enabled = self._as_boolean(
            context.get("integrationValidationEnabled", False)
        )
        if enabled is True:
            checks.append("integration_validation_explicitly_enabled")
        else:
            errors.append(
                "integration_validation_disabled"
                if enabled is False
                else "integration_validation_flag_invalid"
            )

        try:
            client = self._config_resolver(context)
        except (TypeError, ValueError):
            errors.append("existing_indexing_configuration_invalid")

        if client is not None:
            self._validate_client(client, checks, errors)
            self._validate_indexing(client, checks, errors)

        return IntegrationPreflightReport(
            ready=not errors,
            enabled=enabled is True,
            client_id=client.client_id if client is not None else None,
            environment=client.environment if client is not None else None,
            checks=tuple(checks),
            errors=tuple(dict.fromkeys(errors)),
        )

    def require_ready(self, context: Mapping[str, object]) -> ClientConfig:
        report = self.validate(context)
        if not report.ready:
            raise IntegrationValidationError(
                "Offline integration preflight failed: "
                + ", ".join(report.errors)
            )
        return self._config_resolver(context)

    @staticmethod
    def _validate_client(
        client: ClientConfig,
        checks: list[str],
        errors: list[str],
    ) -> None:
        if client.integration_validation_enabled:
            checks.append("existing_integration_gate_enabled")
        else:
            errors.append("existing_integration_gate_disabled")
        if client.client_id == "internal" and client.environment == "dev":
            checks.extend(("internal_client_scope", "non_production_scope"))
        else:
            errors.append("integration_scope_must_be_internal_dev")

    @staticmethod
    def _validate_indexing(
        client: ClientConfig,
        checks: list[str],
        errors: list[str],
    ) -> None:
        config = client.production_indexing
        if config.enabled:
            checks.append("automatic_indexing_explicitly_enabled")
        else:
            errors.append("automatic_indexing_disabled")
        for label, value, pattern in (
            ("bedrock_arn", config.bedrock_model_arn, _BEDROCK_ARN),
            ("secret_arn", config.qdrant_secret_arn, _SECRET_ARN),
            ("layer_arn", config.dependency_layer_arn, _LAYER_ARN),
        ):
            if isinstance(value, str) and pattern.fullmatch(value):
                checks.append(f"{label}_valid")
            else:
                errors.append(f"{label}_invalid")
        if config.qdrant_endpoint_source != "environment":
            errors.append("offline_preflight_requires_explicit_endpoint")
        elif config.qdrant_url is not None:
            try:
                validate_qdrant_endpoint(
                    config.qdrant_url,
                    production=True,
                    tls_required=True,
                )
            except ValueError:
                errors.append("qdrant_https_endpoint_invalid")
            else:
                checks.append("qdrant_https_endpoint_valid")
        else:
            errors.append("qdrant_https_endpoint_invalid")

        collection = config.qdrant_collection or ""
        expected_scope = f"{client.client_id}_{client.environment}".replace(
            "-", "_"
        )
        if (
            _COLLECTION.fullmatch(collection)
            and expected_scope in collection.casefold()
        ):
            checks.append("collection_scope_valid")
        else:
            errors.append("collection_scope_invalid")
        if (
            config.embedding_dimensions is not None
            and 1 <= config.embedding_dimensions <= 4096
        ):
            checks.append("embedding_dimensions_valid")
        else:
            errors.append("embedding_dimensions_invalid")
        for label, value in (
            ("namespace", config.knowledge_namespace),
            ("domain", config.knowledge_domain),
        ):
            if _SAFE_NAME.fullmatch(value):
                checks.append(f"{label}_valid")
            else:
                errors.append(f"{label}_invalid")
        if (
            0 < config.connect_timeout_seconds <= 30
            and 0 < config.request_timeout_seconds <= 300
            and 0 <= config.retry_limit <= 10
            and 0 < config.manifest_conflict_retries <= 10
        ):
            checks.append("timeout_and_retry_bounds_valid")
        else:
            errors.append("timeout_or_retry_bounds_invalid")
        if (
            0 < config.maximum_descriptor_batch_size <= 100
            and 0 < config.maximum_chunks_per_invocation <= 10_000
        ):
            checks.append("work_limits_valid")
        else:
            errors.append("work_limits_invalid")
        if config.qdrant_kms_key_arn is None:
            checks.append("kms_key_not_configured")
        elif _KMS_ARN.fullmatch(config.qdrant_kms_key_arn):
            checks.append("kms_key_arn_valid")
        else:
            errors.append("kms_key_arn_invalid")
        IntegrationPreflightValidator._validate_vpc(config, checks, errors)

    @staticmethod
    def _validate_vpc(config: Any, checks: list[str], errors: list[str]) -> None:
        configured = any(
            (
                config.vpc_id,
                config.subnet_ids,
                config.availability_zones,
                config.qdrant_security_group_id,
            )
        )
        if not configured:
            checks.append("vpc_not_configured")
            return
        valid = (
            isinstance(config.vpc_id, str)
            and re.fullmatch(r"vpc-[a-f0-9]+", config.vpc_id)
            and config.subnet_ids
            and all(re.fullmatch(r"subnet-[a-f0-9]+", item) for item in config.subnet_ids)
            and len(config.subnet_ids) == len(config.availability_zones)
            and all(re.fullmatch(r"[a-z]{2}(?:-gov)?-[a-z]+-\d[a-z]", item) for item in config.availability_zones)
            and isinstance(config.qdrant_security_group_id, str)
            and re.fullmatch(r"sg-[a-f0-9]+", config.qdrant_security_group_id)
        )
        if valid:
            checks.append("vpc_configuration_complete")
        else:
            errors.append("vpc_configuration_incomplete")

    @staticmethod
    def _contains_plaintext_secret(context: Mapping[str, object]) -> bool:
        for key in context:
            normalized = re.sub(r"[^a-z]", "", str(key).casefold())
            if normalized in _PLAINTEXT_SECRET_KEYS:
                return True
        return False

    @staticmethod
    def _as_boolean(value: object) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False
        return None


class IntegrationPhase(StrEnum):
    PREFLIGHT = "preflight"
    READINESS = "readiness"
    UPLOAD_PLAN = "upload_plan"
    INDEXING_OBSERVATION = "indexing_observation"
    VERIFICATION = "verification"
    FAILURE_PATH_VALIDATION = "failure_path_validation"
    CLEANUP_PLAN = "cleanup_plan"


@dataclass(frozen=True)
class IntegrationOperation:
    phase: IntegrationPhase
    operation: str
    requires_connectivity: bool
    state_changing: bool
    billable: bool
    manual_only: bool = False


@dataclass(frozen=True)
class IntegrationValidationPlan:
    client_id: str
    environment: str
    dry_run: bool
    operations: tuple[IntegrationOperation, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "client_id": self.client_id,
            "environment": self.environment,
            "dry_run": self.dry_run,
            "operation_count": len(self.operations),
            "billable_operation_count": sum(
                operation.billable for operation in self.operations
            ),
            "state_changing_operation_count": sum(
                operation.state_changing for operation in self.operations
            ),
            "operations": [
                {
                    "phase": operation.phase.value,
                    "operation": operation.operation,
                    "connectivity": operation.requires_connectivity,
                    "state_changing": operation.state_changing,
                    "billable": operation.billable,
                    "manual_only": operation.manual_only,
                }
                for operation in self.operations
            ],
        }


class IntegrationOperationExecutor(Protocol):
    def execute(
        self,
        operation: IntegrationOperation,
        *,
        client_id: str,
        environment: str,
    ) -> Mapping[str, int | str | bool | None]:
        """Execute one explicitly approved operation."""


class IntegrationValidationRunner:
    """Dry-run-first phase orchestration with an injected live boundary."""

    CONFIRMATION = "APPROVE_INTERNAL_DEV_INTEGRATION"

    def __init__(
        self,
        *,
        preflight: IntegrationPreflightValidator | None = None,
        executor: IntegrationOperationExecutor | None = None,
    ) -> None:
        self._preflight = preflight or IntegrationPreflightValidator()
        self._executor = executor

    def plan(
        self, context: Mapping[str, object]
    ) -> IntegrationValidationPlan:
        client = self._preflight.require_ready(context)
        return IntegrationValidationPlan(
            client_id=client.client_id,
            environment=client.environment,
            dry_run=True,
            operations=self._operations(),
        )

    def execute(
        self,
        context: Mapping[str, object],
        *,
        apply: bool = False,
        confirmation: str | None = None,
        phases: Sequence[IntegrationPhase] | None = None,
    ) -> tuple[Mapping[str, int | str | bool | None], ...]:
        plan = self.plan(context)
        if not apply:
            return ()
        if confirmation != self.CONFIRMATION:
            raise IntegrationValidationError(
                "State-changing integration validation requires confirmation"
            )
        if self._executor is None:
            raise IntegrationValidationError(
                "No deployed integration executor was configured"
            )
        selected = set(phases or IntegrationPhase)
        results = []
        for operation in plan.operations:
            if operation.phase not in selected or operation.manual_only:
                continue
            result = self._executor.execute(
                operation,
                client_id=plan.client_id,
                environment=plan.environment,
            )
            self._validate_safe_result(result)
            results.append(result)
        return tuple(results)

    @staticmethod
    def _operations() -> tuple[IntegrationOperation, ...]:
        return (
            IntegrationOperation(IntegrationPhase.PREFLIGHT, "validate_configuration", False, False, False),
            IntegrationOperation(IntegrationPhase.READINESS, "inspect_deployed_configuration", True, False, True),
            IntegrationOperation(IntegrationPhase.UPLOAD_PLAN, "upload_synthetic_fixture", True, True, True),
            IntegrationOperation(IntegrationPhase.INDEXING_OBSERVATION, "observe_descriptor_and_manifest", True, False, True),
            IntegrationOperation(IntegrationPhase.VERIFICATION, "verify_scoped_vector_and_retrieval", True, False, True),
            IntegrationOperation(IntegrationPhase.FAILURE_PATH_VALIDATION, "inject_controlled_retryable_failure", True, True, True),
            IntegrationOperation(IntegrationPhase.CLEANUP_PLAN, "review_targeted_cleanup", True, True, True, True),
        )

    @staticmethod
    def _validate_safe_result(
        result: Mapping[str, int | str | bool | None]
    ) -> None:
        forbidden = re.compile(
            r"(?:secret|credential|api.?key|vector|content|prompt|text)",
            re.IGNORECASE,
        )
        if any(forbidden.search(str(key)) for key in result):
            raise IntegrationValidationError(
                "Integration executor returned a sensitive field"
            )
        if any(isinstance(value, str) and len(value) > 256 for value in result.values()):
            raise IntegrationValidationError(
                "Integration executor returned an unbounded value"
            )


class IntegrationFailureDisposition(StrEnum):
    RETRYABLE = "retryable"
    PERMANENT = "permanent"
    OPERATOR_ACTION = "operator_action"


_FAILURE_DISPOSITIONS = {
    "IndexingSecretSchemaError": IntegrationFailureDisposition.PERMANENT,
    "AuthenticationError": IntegrationFailureDisposition.OPERATOR_ACTION,
    "TlsError": IntegrationFailureDisposition.OPERATOR_ACTION,
    "VectorStoreUnavailableError": IntegrationFailureDisposition.RETRYABLE,
    "EmbeddingAccessDeniedError": IntegrationFailureDisposition.OPERATOR_ACTION,
    "EmbeddingThrottledError": IntegrationFailureDisposition.RETRYABLE,
    "VectorDimensionMismatchError": IntegrationFailureDisposition.PERMANENT,
    "VectorCollectionConfigurationError": IntegrationFailureDisposition.PERMANENT,
    "ManifestWriteConflictError": IntegrationFailureDisposition.RETRYABLE,
    "ConditionalStorageConflictError": IntegrationFailureDisposition.RETRYABLE,
    "AutomaticIndexingIncompleteError": IntegrationFailureDisposition.RETRYABLE,
    "PartialIndexingRetry": IntegrationFailureDisposition.RETRYABLE,
    "RetriesExhaustedDlqDelivery": IntegrationFailureDisposition.OPERATOR_ACTION,
    "PermanentFailureResetAttempt": IntegrationFailureDisposition.PERMANENT,
    "CrossClientRedriveAttempt": IntegrationFailureDisposition.PERMANENT,
    "SecretRotationRequired": IntegrationFailureDisposition.OPERATOR_ACTION,
}


def classify_integration_failure(
    error_type: str,
) -> IntegrationFailureDisposition:
    """Classify known deployed failure scenarios without raw error payloads."""

    return _FAILURE_DISPOSITIONS.get(
        error_type,
        IntegrationFailureDisposition.OPERATOR_ACTION,
    )


@dataclass(frozen=True)
class IntegrationEvidenceExpectation:
    client_id: str
    environment: str
    namespace: str
    domain: str
    collection: str
    embedding_dimensions: int
    expected_chunk_count: int
    expected_prefix: str = "knowledge/raw/integration/"
    expect_complete: bool = True
    expect_dlq_delivery: bool = False
    forbidden_log_phrases: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntegrationEvidence:
    s3_event_delivered: bool
    descriptor: Mapping[str, Any]
    manifest_entry: Mapping[str, Any]
    vector_payload_summaries: tuple[Mapping[str, Any], ...]
    touched_scopes: tuple[tuple[str, str], ...]
    touched_collections: tuple[str, ...]
    touched_prefixes: tuple[str, ...]
    dlq_message_count: int
    permanent_failure_dispatch_count: int
    log_records: tuple[str, ...]


@dataclass(frozen=True)
class IntegrationEvidenceReport:
    valid: bool
    checks: tuple[str, ...]
    errors: tuple[str, ...]


class IntegrationEvidenceSource(Protocol):
    """Live boundary for collecting sanitized identifier/count evidence."""

    def collect(
        self, expectation: IntegrationEvidenceExpectation
    ) -> IntegrationEvidence:
        """Collect evidence without returning text, vectors, or credentials."""


class IntegrationEvidenceVerifier:
    """Verify sanitized deployed evidence supplied through an adapter."""

    def verify(
        self,
        evidence: IntegrationEvidence,
        expectation: IntegrationEvidenceExpectation,
    ) -> IntegrationEvidenceReport:
        checks: list[str] = []
        errors: list[str] = []
        descriptor = evidence.descriptor
        manifest = evidence.manifest_entry
        if evidence.s3_event_delivered:
            checks.append("s3_event_delivered")
        else:
            errors.append("s3_event_not_observed")
        expected_scope = (expectation.client_id, expectation.environment)
        if (
            descriptor.get("schema_version") == 2
            and descriptor.get("client_id") == expectation.client_id
            and descriptor.get("environment") == expectation.environment
            and descriptor.get("namespace") == expectation.namespace
            and descriptor.get("domain") == expectation.domain
        ):
            checks.append("descriptor_scope_and_schema_valid")
        else:
            errors.append("descriptor_scope_or_schema_invalid")
        states = descriptor.get("chunks")
        if isinstance(states, list) and len(states) == expectation.expected_chunk_count:
            indexed = sum(
                isinstance(state, Mapping) and state.get("status") == "indexed"
                for state in states
            )
            if (
                expectation.expect_complete
                and indexed == len(states)
                and descriptor.get("index_status") == "complete"
            ):
                checks.append("descriptor_indexing_complete")
            elif not expectation.expect_complete and all(
                isinstance(state, Mapping)
                and state.get("status") in {"pending", "indexed"}
                for state in states
            ):
                checks.append("descriptor_resumable_state_valid")
            else:
                errors.append("descriptor_chunk_state_invalid")
        else:
            errors.append("descriptor_chunk_count_invalid")
        if (
            descriptor.get("vector_dimension") == expectation.embedding_dimensions
            and descriptor.get("vector_collection") == expectation.collection
        ):
            checks.append("descriptor_vector_contract_valid")
        else:
            errors.append("descriptor_vector_contract_invalid")
        if (
            manifest.get("vector_dimension") == expectation.embedding_dimensions
            and manifest.get("vector_collection") == expectation.collection
            and manifest.get("index_status") == descriptor.get("index_status")
        ):
            checks.append("manifest_indexing_fields_valid")
        else:
            errors.append("manifest_indexing_fields_invalid")
        if evidence.touched_scopes and set(evidence.touched_scopes) == {expected_scope}:
            checks.append("no_cross_scope_touch")
        else:
            errors.append("cross_scope_touch_detected")
        if (
            evidence.touched_collections
            and set(evidence.touched_collections) == {expectation.collection}
        ):
            checks.append("no_cross_collection_touch")
        else:
            errors.append("cross_collection_touch_detected")
        if evidence.touched_prefixes and all(
            prefix.startswith(expectation.expected_prefix)
            for prefix in evidence.touched_prefixes
        ):
            checks.append("no_cross_prefix_touch")
        else:
            errors.append("cross_prefix_touch_detected")
        if evidence.vector_payload_summaries and all(
            payload.get("client_id") == expectation.client_id
            and payload.get("environment") == expectation.environment
            and payload.get("namespace") == expectation.namespace
            and payload.get("domain") == expectation.domain
            and payload.get("collection") == expectation.collection
            and payload.get("dimensions") == expectation.embedding_dimensions
            for payload in evidence.vector_payload_summaries
        ):
            checks.append("vector_payload_scope_valid")
        else:
            errors.append("vector_payload_scope_invalid")
        if evidence.permanent_failure_dispatch_count == 0:
            checks.append("permanent_failures_not_redriven")
        else:
            errors.append("permanent_failure_was_redriven")
        if (
            evidence.dlq_message_count > 0
            if expectation.expect_dlq_delivery
            else evidence.dlq_message_count >= 0
        ):
            checks.append("dlq_evidence_valid")
        else:
            errors.append("dlq_evidence_count_invalid")
        forbidden_log_terms = (
            "api_key",
            "embedding_vector",
            "inputText",
            *expectation.forbidden_log_phrases,
        )
        if not any(
            term in record for record in evidence.log_records
            for term in forbidden_log_terms
        ):
            checks.append("logs_use_safe_fields")
        else:
            errors.append("sensitive_log_field_detected")
        return IntegrationEvidenceReport(
            valid=not errors,
            checks=tuple(checks),
            errors=tuple(errors),
        )
