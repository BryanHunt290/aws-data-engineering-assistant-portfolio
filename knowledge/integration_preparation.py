"""Offline operator preparation for internal-dev indexing validation.

This module handles local files and synthesized templates only. It deliberately
contains no AWS SDK, Bedrock, Qdrant, Docker, or CDK execution boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
from urllib.parse import urlparse

from config.clients import build_stack_id
from knowledge.integration_validation import (
    IntegrationPreflightReport,
    IntegrationPreflightValidator,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXAMPLE_PATH = (
    REPOSITORY_ROOT
    / "config"
    / "integration-validation.internal-dev.example.json"
)
DEFAULT_LOCAL_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "config"
    / "integration-validation.internal-dev.local.json"
)
DEFAULT_CONTEXT_PATH = (
    REPOSITORY_ROOT / ".local" / "internal-dev-indexing-context.ps1"
)
DEFAULT_AUDIT_PATH = (
    REPOSITORY_ROOT / ".local" / "internal-dev-indexing-audit.json"
)
DEFAULT_EXPECTATIONS_PATH = (
    REPOSITORY_ROOT / ".local" / "internal-dev-expected-resources.json"
)

REQUIRED_NON_SECRET_FIELDS = (
    "client",
    "clientId",
    "environment",
    "awsRegion",
    "automaticIndexingEnabled",
    "integrationValidationEnabled",
    "indexingEmbeddingProvider",
    "indexingEmbeddingModelId",
    "indexingEmbeddingDimensions",
    "indexingBedrockModelArn",
    "indexingVectorStoreProvider",
    "indexingQdrantEndpointSource",
    "indexingQdrantUrl",
    "indexingQdrantCollection",
    "indexingDependencyLayerArn",
    "indexingKnowledgeNamespace",
    "indexingKnowledgeDomain",
    "indexingConnectTimeoutSeconds",
    "indexingRequestTimeoutSeconds",
    "indexingRetryLimit",
    "indexingManifestConflictRetries",
    "indexingMaximumDescriptorBatchSize",
    "indexingMaximumChunksPerInvocation",
)
SECRET_REFERENCE_FIELDS = (
    "indexingQdrantSecretArn",
)
OPTIONAL_NON_SECRET_FIELDS = (
    "indexingQdrantKmsKeyArn",
    "indexingReservedConcurrentExecutions",
)
OPTIONAL_VPC_FIELDS = (
    "indexingVpcId",
    "indexingSubnetIds",
    "indexingAvailabilityZones",
    "indexingQdrantSecurityGroupId",
)
PROHIBITED_PLAINTEXT_FIELDS = (
    "appQdrantApiKey",
    "indexingQdrantApiKey",
    "knowledgeQdrantApiKey",
    "qdrantApiKey",
)

_BEDROCK_ARN = re.compile(
    r"arn:(?:aws|aws-us-gov):bedrock:([a-z0-9-]+):(?:\d{12})?:"
    r"(?:foundation-model|inference-profile|application-inference-profile)/"
    r"[A-Za-z0-9._:/+-]+"
)
_ACCOUNT_ARN = {
    "indexingQdrantSecretArn": re.compile(
        r"arn:(?:aws|aws-us-gov):secretsmanager:([a-z0-9-]+):\d{12}:"
        r"secret:[A-Za-z0-9/_+=.@-]+"
    ),
    "indexingDependencyLayerArn": re.compile(
        r"arn:(?:aws|aws-us-gov):lambda:([a-z0-9-]+):\d{12}:"
        r"layer:[A-Za-z0-9_-]+:[1-9]\d*"
    ),
    "indexingQdrantKmsKeyArn": re.compile(
        r"arn:(?:aws|aws-us-gov):kms:([a-z0-9-]+):\d{12}:"
        r"key/[A-Za-z0-9-]+"
    ),
}
_REGION = re.compile(r"[a-z]{2}(?:-gov)?-[a-z]+-\d")
_SAFE_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")
_COLLECTION = re.compile(r"[a-z0-9](?:[a-z0-9_]{0,118}[a-z0-9])?")
_VPC = re.compile(r"vpc-[a-f0-9]+")
_SUBNET = re.compile(r"subnet-[a-f0-9]+")
_SECURITY_GROUP = re.compile(r"sg-[a-f0-9]+")


class FieldStatus(StrEnum):
    READY = "ready"
    PLACEHOLDER = "placeholder"
    MISSING = "missing"
    INVALID = "invalid"
    OPTIONAL_NOT_CONFIGURED = "optional-not-configured"
    PROHIBITED = "prohibited"


@dataclass(frozen=True)
class ConfigurationFieldReview:
    field: str
    category: str
    status: FieldStatus
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "field": self.field,
            "category": self.category,
            "status": self.status.value,
            "message": self.message,
        }


@dataclass(frozen=True)
class ConfigurationReviewReport:
    ready: bool
    fields: tuple[ConfigurationFieldReview, ...]
    preflight: IntegrationPreflightReport

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "field_count": len(self.fields),
            "fields": [field.to_dict() for field in self.fields],
            "preflight": self.preflight.to_dict(),
            "network_calls": 0,
        }


@dataclass(frozen=True)
class TemplateReviewReport:
    valid: bool
    checks: tuple[str, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Configuration must be a JSON object")
    return payload


def require_repository_path(
    path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    root = repository_root.resolve()
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("Path must remain inside the repository")
    return resolved


def require_generated_output_path(
    path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    resolved = require_repository_path(path, repository_root=repository_root)
    allowed_roots = (
        (repository_root / ".local").resolve(),
        (repository_root / "build").resolve(),
    )
    if not any(resolved.is_relative_to(root) for root in allowed_roots):
        raise ValueError("Generated artifacts must be under .local or build")
    return resolved


def is_expected_ignored_path(
    path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> bool:
    resolved = require_repository_path(path, repository_root=repository_root)
    relative = resolved.relative_to(repository_root.resolve())
    if relative.parts[0] in {".local", "build"}:
        return True
    return (
        len(relative.parts) == 2
        and relative.parts[0] == "config"
        and relative.name.endswith(".local.json")
    )


def bootstrap_local_configuration(
    *,
    source: Path = DEFAULT_EXAMPLE_PATH,
    output: Path = DEFAULT_LOCAL_CONFIG_PATH,
    force: bool = False,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    source_path = require_repository_path(source, repository_root=repository_root)
    output_path = require_repository_path(output, repository_root=repository_root)
    if not is_expected_ignored_path(output_path, repository_root=repository_root):
        raise ValueError("Local configuration output must match an ignored path")
    if (repository_root / ".git").exists():
        relative = output_path.relative_to(repository_root.resolve()).as_posix()
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", relative],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if ignored.returncode != 0:
            raise ValueError("Local configuration output is not ignored by Git")
    if output_path.exists() and not force:
        raise FileExistsError("Local configuration already exists; use --force")
    payload = load_json_object(source_path)
    for key in PROHIBITED_PLAINTEXT_FIELDS:
        if key in payload:
            raise ValueError("Example configuration contains a prohibited field")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def review_configuration(
    context: Mapping[str, object],
    *,
    preflight: IntegrationPreflightValidator | None = None,
) -> ConfigurationReviewReport:
    fields: list[ConfigurationFieldReview] = []

    def add(
        name: str,
        status: FieldStatus,
        message: str,
        category: str = "required-non-secret",
    ) -> None:
        fields.append(ConfigurationFieldReview(name, category, status, message))

    _review_exact(context, "client", "internal-dev", add)
    _review_exact(context, "clientId", "internal", add)
    _review_exact(context, "environment", "dev", add)
    _review_region(context, add)
    _review_true_flag(context, "automaticIndexingEnabled", add)
    _review_true_flag(context, "integrationValidationEnabled", add)
    add(
        "knowledgeBucketReference",
        FieldStatus.READY,
        "stack-managed KnowledgeBucket reference",
        "derived-non-secret",
    )
    _review_exact(context, "indexingEmbeddingProvider", "bedrock", add)
    _review_nonempty(context, "indexingEmbeddingModelId", add)
    _review_integer(context, "indexingEmbeddingDimensions", 1, 4096, add)
    _review_arn(context, "indexingBedrockModelArn", _BEDROCK_ARN, add)
    _review_exact(context, "indexingVectorStoreProvider", "qdrant", add)
    _review_exact(context, "indexingQdrantEndpointSource", "environment", add)
    _review_https_endpoint(context, add)
    _review_collection(context, add)
    _review_arn(
        context,
        "indexingQdrantSecretArn",
        _ACCOUNT_ARN["indexingQdrantSecretArn"],
        add,
        category="secret-reference",
    )
    _review_arn(context, "indexingDependencyLayerArn", _ACCOUNT_ARN["indexingDependencyLayerArn"], add)
    _review_optional_arn(context, "indexingQdrantKmsKeyArn", add)
    _review_safe_name(context, "indexingKnowledgeNamespace", add)
    _review_safe_name(context, "indexingKnowledgeDomain", add)
    _review_number(context, "indexingConnectTimeoutSeconds", 0, 30, add)
    _review_number(context, "indexingRequestTimeoutSeconds", 0, 300, add)
    _review_integer(context, "indexingRetryLimit", 0, 10, add)
    _review_integer(context, "indexingManifestConflictRetries", 1, 10, add)
    _review_integer(context, "indexingMaximumDescriptorBatchSize", 1, 100, add)
    _review_integer(context, "indexingMaximumChunksPerInvocation", 1, 10_000, add)
    _review_optional_positive_integer(
        context, "indexingReservedConcurrentExecutions", add
    )
    _review_vpc(context, add)
    prohibited_names = {
        _normalized_key(key) for key in PROHIBITED_PLAINTEXT_FIELDS
    }
    found_plaintext = [
        str(key)
        for key in context
        if _normalized_key(str(key)) in prohibited_names
    ]
    if found_plaintext:
        for key in found_plaintext:
            add(key, FieldStatus.PROHIBITED, "plaintext credential field must be removed", "prohibited-plaintext-secret")
    else:
        add(
            "plaintextApiKeyFields",
            FieldStatus.READY,
            "prohibited plaintext credential fields are absent",
            "prohibited-plaintext-secret",
        )

    validator = preflight or IntegrationPreflightValidator()
    preflight_report = validator.validate(context)
    blocking = {
        FieldStatus.PLACEHOLDER,
        FieldStatus.MISSING,
        FieldStatus.INVALID,
        FieldStatus.PROHIBITED,
    }
    return ConfigurationReviewReport(
        ready=preflight_report.ready and not any(field.status in blocking for field in fields),
        fields=tuple(fields),
        preflight=preflight_report,
    )


def normalized_non_secret_configuration(
    context: Mapping[str, object],
) -> dict[str, object]:
    prohibited = {_normalized_key(key) for key in PROHIBITED_PLAINTEXT_FIELDS}
    result = {
        str(key): _normalize_json_value(value)
        for key, value in sorted(context.items(), key=lambda item: str(item[0]))
        if _normalized_key(str(key)) not in prohibited
    }
    return result


def configuration_fingerprint(context: Mapping[str, object]) -> str:
    normalized = json.dumps(
        normalized_non_secret_configuration(context),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def generate_context_artifacts(
    context: Mapping[str, object],
    *,
    config_path: Path,
    context_output: Path = DEFAULT_CONTEXT_PATH,
    audit_output: Path = DEFAULT_AUDIT_PATH,
    force: bool = False,
    repository_root: Path = REPOSITORY_ROOT,
    timestamp: datetime | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    report = review_configuration(context)
    if not report.ready:
        raise ValueError("Configuration must pass offline review before context generation")
    config_path = require_repository_path(config_path, repository_root=repository_root)
    context_path = require_generated_output_path(context_output, repository_root=repository_root)
    audit_path = require_generated_output_path(audit_output, repository_root=repository_root)
    for path in (context_path, audit_path):
        if path.exists() and not force:
            raise FileExistsError(f"Generated artifact already exists: {path.name}")

    lines = [
        "# GENERATED OFFLINE: reviewed internal-dev production-indexing context.",
        "# SAFE ONLY FOR: cdk synth --no-lookups",
        "# NOT APPROVED FOR: cdk diff or cdk deploy",
        "$ContextArgs = @(",
    ]
    for key, value in normalized_non_secret_configuration(context).items():
        rendered = _context_value(value).replace("'", "''")
        lines.extend(("    '-c'", f"    '{key}={rendered}'"))
    lines.extend((")", ""))

    context_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text("\n".join(lines), encoding="utf-8")
    audit = build_audit_artifact(
        context,
        config_path=config_path,
        context_path=context_path,
        preflight=report.preflight,
        repository_root=repository_root,
        timestamp=timestamp,
    )
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return context_path, audit_path, audit


def build_audit_artifact(
    context: Mapping[str, object],
    *,
    config_path: Path,
    context_path: Path,
    preflight: IntegrationPreflightReport,
    repository_root: Path = REPOSITORY_ROOT,
    timestamp: datetime | None = None,
) -> dict[str, object]:
    commit, dirty = _repository_state(repository_root)
    client_id = str(context["clientId"])
    environment = str(context["environment"])
    return {
        "schema_version": 1,
        "timestamp": (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(),
        "repository_commit": commit,
        "repository_state": "dirty" if dirty else "clean",
        "python_version": ".".join(str(item) for item in sys.version_info[:3]),
        "configuration_fingerprint_sha256": configuration_fingerprint(context),
        "local_configuration_path": _relative_display(config_path, repository_root),
        "preflight": preflight.to_dict(),
        "generated_context_artifact_path": _relative_display(context_path, repository_root),
        "expected_collection": context["indexingQdrantCollection"],
        "expected_client_id": client_id,
        "expected_environment": environment,
        "expected_namespace": context["indexingKnowledgeNamespace"],
        "expected_domain": context["indexingKnowledgeDomain"],
        "expected_embedding_dimensions": context["indexingEmbeddingDimensions"],
        "expected_stack_name": build_stack_id(client_id, environment),
        "secret_values_recorded": False,
        "network_calls": 0,
    }


def expected_resource_report(
    context: Mapping[str, object],
    *,
    template: Mapping[str, object] | None = None,
) -> dict[str, object]:
    review = review_configuration(context)
    if not review.ready:
        raise ValueError("Configuration must pass offline review before resource review")
    vpc_enabled = bool(context.get("indexingVpcId"))
    expectations = (
        "document_ingestion_lambda_python_3_12",
        "versioned_dependency_layer_attached",
        "bedrock_invoke_model_scoped_to_configured_arn",
        "secretsmanager_get_secret_value_scoped_to_configured_arn",
        "optional_kms_decrypt_scoped_to_configured_arn",
        "optional_vpc_attachment_matches_reviewed_identifiers",
        "private_vpc_has_secrets_logs_bedrock_and_s3_interface_endpoints",
        "lambda_and_endpoint_security_groups_are_separate",
        "lambda_egress_to_qdrant_security_group_is_tcp_443_only",
        "reserved_concurrency_matches_optional_configuration",
        "encrypted_dead_letter_queue_is_attached",
        "s3_object_created_notification_is_raw_prefix_only",
        "indexing_environment_is_scoped_and_bounded",
        "plaintext_qdrant_credentials_are_absent",
        "qdrant_infrastructure_is_not_created",
        "production_client_resources_are_absent",
    )
    template_review = (
        review_synthesized_template(template, context).to_dict()
        if template is not None
        else {"valid": None, "checks": [], "errors": ["template_not_supplied"]}
    )
    return {
        "schema_version": 1,
        "scope": {"client_id": "internal", "environment": "dev"},
        "stack_name": build_stack_id("internal", "dev"),
        "vpc_expected": vpc_enabled,
        "expectations": list(expectations),
        "template_review": template_review,
        "safe_for": ["cdk synth --no-lookups"],
        "not_approved_for": ["cdk diff", "cdk deploy"],
        "network_calls": 0,
    }


def review_synthesized_template(
    template: Mapping[str, object],
    context: Mapping[str, object],
) -> TemplateReviewReport:
    resources = template.get("Resources")
    if not isinstance(resources, Mapping):
        return TemplateReviewReport(False, (), ("template_resources_missing",))
    checks: list[str] = []
    errors: list[str] = []

    functions = _resources_of_type(resources, "AWS::Lambda::Function")
    ingestion = next(
        (
            resource
            for resource in functions
            if _nested(resource, "Properties", "Handler")
            == "lambda.document_ingestion.index.handler"
        ),
        None,
    )
    if ingestion is None:
        errors.append("document_ingestion_lambda_missing")
    else:
        properties = ingestion.get("Properties", {})
        _check(properties.get("Runtime") == "python3.12", "lambda_runtime_valid", "lambda_runtime_invalid", checks, errors)
        configured_concurrency = context.get(
            "indexingReservedConcurrentExecutions"
        )
        expected_concurrency = (
            None
            if configured_concurrency in {None, ""}
            else int(configured_concurrency)
        )
        _check(
            properties.get("ReservedConcurrentExecutions")
            == expected_concurrency,
            "reserved_concurrency_matches_configuration",
            "reserved_concurrency_invalid",
            checks,
            errors,
        )
        _check(
            str(context["indexingDependencyLayerArn"])
            in json.dumps(properties.get("Layers", []), sort_keys=True),
            "dependency_layer_attached",
            "dependency_layer_missing_or_incorrect",
            checks,
            errors,
        )
        _check(bool(properties.get("DeadLetterConfig")), "dead_letter_queue_attached", "dead_letter_queue_missing", checks, errors)
        variables = _nested(properties, "Environment", "Variables") or {}
        required_environment = {
            "CLIENT_ID": "internal",
            "DEPLOYMENT_ENVIRONMENT": "dev",
            "KNOWLEDGE_AUTOMATIC_INDEXING_ENABLED": "true",
            "KNOWLEDGE_INDEXING_RUNTIME_MODE": "production",
            "KNOWLEDGE_EMBEDDING_PROVIDER": context["indexingEmbeddingProvider"],
            "KNOWLEDGE_EMBEDDING_MODEL_ID": context["indexingEmbeddingModelId"],
            "KNOWLEDGE_EMBEDDING_DIMENSIONS": str(context["indexingEmbeddingDimensions"]),
            "KNOWLEDGE_VECTOR_STORE_PROVIDER": context["indexingVectorStoreProvider"],
            "KNOWLEDGE_QDRANT_ENDPOINT_SOURCE": context["indexingQdrantEndpointSource"],
            "KNOWLEDGE_QDRANT_URL": context["indexingQdrantUrl"],
            "KNOWLEDGE_QDRANT_COLLECTION": context["indexingQdrantCollection"],
            "KNOWLEDGE_QDRANT_SECRET_IDENTIFIER": context["indexingQdrantSecretArn"],
            "KNOWLEDGE_QDRANT_TLS_REQUIRED": "true",
            "KNOWLEDGE_QDRANT_AUTHENTICATION_REQUIRED": "true",
            "KNOWLEDGE_CONNECT_TIMEOUT_SECONDS": str(float(context["indexingConnectTimeoutSeconds"])),
            "KNOWLEDGE_REQUEST_TIMEOUT_SECONDS": str(float(context["indexingRequestTimeoutSeconds"])),
            "KNOWLEDGE_INDEXING_RETRY_LIMIT": str(context["indexingRetryLimit"]),
            "KNOWLEDGE_MANIFEST_CONFLICT_RETRIES": str(context["indexingManifestConflictRetries"]),
            "KNOWLEDGE_MAX_DESCRIPTOR_BATCH_SIZE": str(context["indexingMaximumDescriptorBatchSize"]),
            "KNOWLEDGE_MAX_CHUNKS_PER_INVOCATION": str(context["indexingMaximumChunksPerInvocation"]),
            "KNOWLEDGE_NAMESPACE": context["indexingKnowledgeNamespace"],
            "KNOWLEDGE_DOMAIN": context["indexingKnowledgeDomain"],
        }
        _check(
            isinstance(variables, Mapping) and all(variables.get(key) == value for key, value in required_environment.items()),
            "indexing_environment_valid",
            "indexing_environment_invalid",
            checks,
            errors,
        )
        vpc_expected = bool(context.get("indexingVpcId"))
        _check(
            bool(properties.get("VpcConfig")) == vpc_expected,
            "optional_vpc_attachment_valid",
            "optional_vpc_attachment_invalid",
            checks,
            errors,
        )
        _check(
            isinstance(variables, Mapping)
            and isinstance(variables.get("KNOWLEDGE_BUCKET_NAME"), Mapping)
            and "Ref" in variables["KNOWLEDGE_BUCKET_NAME"],
            "knowledge_bucket_reference_valid",
            "knowledge_bucket_reference_invalid",
            checks,
            errors,
        )

    serialized = json.dumps(template, sort_keys=True)
    for action, resource, success, failure in (
        ("bedrock:InvokeModel", context["indexingBedrockModelArn"], "bedrock_permission_scoped", "bedrock_permission_missing_or_unscoped"),
        ("secretsmanager:GetSecretValue", context["indexingQdrantSecretArn"], "secret_permission_scoped", "secret_permission_missing_or_unscoped"),
    ):
        _check(_policy_pair_present(resources, action, resource), success, failure, checks, errors)
    kms_arn = context.get("indexingQdrantKmsKeyArn")
    _check(
        _policy_pair_present(resources, "kms:Decrypt", kms_arn) if kms_arn else "kms:Decrypt" not in serialized,
        "optional_kms_permission_valid",
        "optional_kms_permission_invalid",
        checks,
        errors,
    )
    queues = _resources_of_type(resources, "AWS::SQS::Queue")
    _check(
        any(_nested(queue, "Properties", "SqsManagedSseEnabled") is True for queue in queues),
        "dead_letter_queue_encrypted",
        "dead_letter_queue_encryption_missing",
        checks,
        errors,
    )
    _check(
        any(_nested(queue, "Properties", "MessageRetentionPeriod") == 1_209_600 for queue in queues)
        and "aws:SecureTransport" in serialized,
        "dead_letter_queue_behavior_bounded",
        "dead_letter_queue_behavior_invalid",
        checks,
        errors,
    )
    buckets = [
        resource
        for resource in resources.values()
        if isinstance(resource, Mapping)
        and resource.get("Type")
        in {"AWS::S3::Bucket", "Custom::S3BucketNotifications"}
    ]
    _check(
        any(_raw_prefix_notification(bucket) for bucket in buckets),
        "s3_raw_prefix_notification_valid",
        "s3_raw_prefix_notification_missing",
        checks,
        errors,
    )
    vpc_expected = bool(context.get("indexingVpcId"))
    endpoints = _resources_of_type(resources, "AWS::EC2::VPCEndpoint")
    security_groups = _resources_of_type(resources, "AWS::EC2::SecurityGroup")
    if vpc_expected:
        endpoint_text = json.dumps(endpoints, sort_keys=True).casefold()
        _check(
            len(endpoints) == 4
            and all(
                service in endpoint_text
                for service in ("secretsmanager", "logs", "bedrock-runtime", "s3")
            ),
            "required_interface_endpoints_present",
            "required_interface_endpoints_invalid",
            checks,
            errors,
        )
        _check(len(security_groups) >= 2, "lambda_and_endpoint_security_groups_present", "indexing_security_groups_missing", checks, errors)
        egress_rules = _resources_of_type(resources, "AWS::EC2::SecurityGroupEgress")
        _check(
            any(
                _nested(rule, "Properties", "DestinationSecurityGroupId")
                == context["indexingQdrantSecurityGroupId"]
                and _nested(rule, "Properties", "IpProtocol") == "tcp"
                and _nested(rule, "Properties", "FromPort") == 443
                and _nested(rule, "Properties", "ToPort") == 443
                for rule in egress_rules
            ),
            "qdrant_egress_rule_present",
            "qdrant_egress_rule_missing",
            checks,
            errors,
        )
        _check(
            str(context["indexingVpcId"]) in serialized
            and all(str(item) in serialized for item in _string_sequence(context["indexingSubnetIds"])),
            "reviewed_vpc_identifiers_present",
            "reviewed_vpc_identifiers_missing",
            checks,
            errors,
        )
    else:
        _check(not endpoints and not security_groups, "vpc_resources_absent_when_optional", "unexpected_vpc_resources", checks, errors)
    lowered = serialized.casefold()
    _check(
        not any(_normalized_key(key) in _normalized_key(lowered) for key in PROHIBITED_PLAINTEXT_FIELDS),
        "plaintext_credentials_absent",
        "plaintext_credentials_present",
        checks,
        errors,
    )
    _check("AWS::Qdrant" not in serialized, "qdrant_infrastructure_absent", "qdrant_infrastructure_present", checks, errors)
    scoped_environments = [
        _nested(function, "Properties", "Environment", "Variables")
        for function in functions
    ]
    _check(
        all(
            not isinstance(environment, Mapping)
            or (
                environment.get("CLIENT_ID") in {None, "internal"}
                and environment.get("DEPLOYMENT_ENVIRONMENT") in {None, "dev"}
            )
            for environment in scoped_environments
        ),
        "production_client_resources_absent",
        "production_client_resource_leakage",
        checks,
        errors,
    )
    return TemplateReviewReport(not errors, tuple(checks), tuple(errors))


def write_expected_resource_report(
    context: Mapping[str, object],
    *,
    output: Path,
    template: Mapping[str, object] | None = None,
    force: bool = False,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[Path, dict[str, object]]:
    path = require_generated_output_path(output, repository_root=repository_root)
    if path.exists() and not force:
        raise FileExistsError("Expected-resource report already exists; use --force")
    report = expected_resource_report(context, template=template)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, report


def _review_exact(context: Mapping[str, object], key: str, expected: str, add: Any) -> None:
    value = context.get(key)
    if value is None or value == "":
        add(key, FieldStatus.MISSING, "required field is missing")
    elif not isinstance(value, str) or value.strip().casefold() != expected.casefold():
        add(key, FieldStatus.INVALID, f"must identify {expected}")
    else:
        add(key, FieldStatus.READY, "reviewed identifier is valid")


def _review_true_flag(context: Mapping[str, object], key: str, add: Any) -> None:
    value = context.get(key)
    normalized = value if isinstance(value, bool) else str(value).strip().casefold() if value is not None else None
    if value is None:
        add(key, FieldStatus.MISSING, "explicit opt-in flag is missing")
    elif normalized is True or normalized in {"true", "1", "yes"}:
        add(key, FieldStatus.READY, "explicit opt-in is enabled")
    else:
        add(key, FieldStatus.INVALID, "must be explicitly true")


def _review_region(context: Mapping[str, object], add: Any) -> None:
    value = context.get("awsRegion")
    if value is None or value == "":
        add("awsRegion", FieldStatus.MISSING, "explicit AWS Region is missing")
    elif not isinstance(value, str) or not _REGION.fullmatch(value.strip().lower()):
        add("awsRegion", FieldStatus.INVALID, "AWS Region identifier is invalid")
    else:
        add("awsRegion", FieldStatus.READY, "AWS Region identifier is valid")


def _review_nonempty(context: Mapping[str, object], key: str, add: Any) -> None:
    value = context.get(key)
    if value is None or value == "":
        add(key, FieldStatus.MISSING, "required field is missing")
    elif not isinstance(value, str) or not value.strip():
        add(key, FieldStatus.INVALID, "must be a non-empty string")
    elif _is_placeholder(value):
        add(key, FieldStatus.PLACEHOLDER, "placeholder must be replaced and reviewed")
    else:
        add(key, FieldStatus.READY, "value is structurally ready")


def _review_arn(context: Mapping[str, object], key: str, pattern: re.Pattern[str], add: Any, category: str = "required-non-secret") -> None:
    value = context.get(key)
    if value is None or value == "":
        add(key, FieldStatus.MISSING, "required ARN is missing", category)
    elif not isinstance(value, str) or pattern.fullmatch(value.strip()) is None:
        add(key, FieldStatus.INVALID, "ARN format is invalid", category)
    elif _is_placeholder(value):
        add(key, FieldStatus.PLACEHOLDER, "synthetic ARN must be replaced", category)
    elif _arn_region(value) != str(context.get("awsRegion", "")):
        add(key, FieldStatus.INVALID, "ARN Region does not match awsRegion", category)
    else:
        add(key, FieldStatus.READY, "ARN reference is structurally ready", category)


def _review_optional_arn(context: Mapping[str, object], key: str, add: Any) -> None:
    value = context.get(key)
    if value is None or value == "":
        add(key, FieldStatus.OPTIONAL_NOT_CONFIGURED, "optional KMS reference is not configured", "optional-non-secret-reference")
    else:
        _review_arn(context, key, _ACCOUNT_ARN[key], add, category="optional-non-secret-reference")


def _review_optional_positive_integer(
    context: Mapping[str, object], key: str, add: Any
) -> None:
    value = context.get(key)
    if value is None or value == "":
        add(
            key,
            FieldStatus.OPTIONAL_NOT_CONFIGURED,
            "optional reserved concurrency is not configured",
            "optional-non-secret",
        )
    else:
        _review_integer(context, key, 1, 10_000, add)


def _review_https_endpoint(context: Mapping[str, object], add: Any) -> None:
    key = "indexingQdrantUrl"
    value = context.get(key)
    if value is None or value == "":
        add(key, FieldStatus.MISSING, "explicit HTTPS endpoint is missing")
        return
    if not isinstance(value, str):
        add(key, FieldStatus.INVALID, "endpoint must be a string")
        return
    parsed = urlparse(value.strip())
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        add(key, FieldStatus.INVALID, "endpoint must be credential-free HTTPS")
    elif _is_placeholder(value):
        add(key, FieldStatus.PLACEHOLDER, "reserved placeholder endpoint must be replaced")
    else:
        add(key, FieldStatus.READY, "HTTPS endpoint is structurally ready")


def _review_collection(context: Mapping[str, object], add: Any) -> None:
    key = "indexingQdrantCollection"
    value = context.get(key)
    expected_scope = f"{context.get('clientId', '')}_{context.get('environment', '')}".replace("-", "_")
    if value is None or value == "":
        add(key, FieldStatus.MISSING, "collection is missing")
    elif not isinstance(value, str) or not _COLLECTION.fullmatch(value) or expected_scope not in value:
        add(key, FieldStatus.INVALID, "collection must be valid and scoped to internal_dev")
    else:
        add(key, FieldStatus.READY, "collection is scoped to internal-dev")


def _review_safe_name(context: Mapping[str, object], key: str, add: Any) -> None:
    value = context.get(key)
    if value is None or value == "":
        add(key, FieldStatus.MISSING, "required scoped name is missing")
    elif not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        add(key, FieldStatus.INVALID, "scoped name format is invalid")
    else:
        add(key, FieldStatus.READY, "scoped name is valid")


def _review_integer(context: Mapping[str, object], key: str, minimum: int, maximum: int, add: Any) -> None:
    value = context.get(key)
    if value is None or value == "":
        add(key, FieldStatus.MISSING, "required integer is missing")
        return
    if isinstance(value, bool):
        add(key, FieldStatus.INVALID, "must be an integer")
        return
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        add(key, FieldStatus.INVALID, "must be an integer")
        return
    if isinstance(value, float) and not value.is_integer() or not minimum <= parsed <= maximum:
        add(key, FieldStatus.INVALID, f"must be between {minimum} and {maximum}")
    else:
        add(key, FieldStatus.READY, "bounded integer is valid")


def _review_number(context: Mapping[str, object], key: str, minimum_exclusive: float, maximum: float, add: Any) -> None:
    value = context.get(key)
    if value is None or value == "":
        add(key, FieldStatus.MISSING, "required numeric limit is missing")
        return
    if isinstance(value, bool):
        add(key, FieldStatus.INVALID, "must be numeric")
        return
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        add(key, FieldStatus.INVALID, "must be numeric")
        return
    if not minimum_exclusive < parsed <= maximum:
        add(key, FieldStatus.INVALID, f"must be greater than {minimum_exclusive} and at most {maximum}")
    else:
        add(key, FieldStatus.READY, "bounded numeric limit is valid")


def _review_vpc(context: Mapping[str, object], add: Any) -> None:
    configured = any(context.get(key) not in {None, "", ()} if not isinstance(context.get(key), list) else bool(context.get(key)) for key in OPTIONAL_VPC_FIELDS)
    if not configured:
        for key in OPTIONAL_VPC_FIELDS:
            add(key, FieldStatus.OPTIONAL_NOT_CONFIGURED, "optional VPC routing is not configured", "optional-vpc-identifier")
        return
    values = {
        "indexingVpcId": context.get("indexingVpcId"),
        "indexingSubnetIds": _string_sequence(context.get("indexingSubnetIds")),
        "indexingAvailabilityZones": _string_sequence(context.get("indexingAvailabilityZones")),
        "indexingQdrantSecurityGroupId": context.get("indexingQdrantSecurityGroupId"),
    }
    valid = {
        "indexingVpcId": isinstance(values["indexingVpcId"], str) and bool(_VPC.fullmatch(values["indexingVpcId"])),
        "indexingSubnetIds": bool(values["indexingSubnetIds"]) and all(_SUBNET.fullmatch(value) for value in values["indexingSubnetIds"]),
        "indexingAvailabilityZones": bool(values["indexingAvailabilityZones"]) and all(_REGION.fullmatch(value[:-1]) and value.startswith(str(context.get("awsRegion", ""))) for value in values["indexingAvailabilityZones"]),
        "indexingQdrantSecurityGroupId": isinstance(values["indexingQdrantSecurityGroupId"], str) and bool(_SECURITY_GROUP.fullmatch(values["indexingQdrantSecurityGroupId"])),
    }
    if len(values["indexingSubnetIds"]) != len(values["indexingAvailabilityZones"]):
        valid["indexingSubnetIds"] = valid["indexingAvailabilityZones"] = False
    for key in OPTIONAL_VPC_FIELDS:
        add(key, FieldStatus.READY if valid[key] else FieldStatus.INVALID, "optional VPC identifier is ready" if valid[key] else "optional VPC configuration is incomplete or invalid", "optional-vpc-identifier")


def _is_placeholder(value: str) -> bool:
    lowered = value.strip().casefold()
    hostname = urlparse(value).hostname
    return (
        "000000000000" in lowered
        or "replace_me" in lowered
        or "replace-me" in lowered
        or "<" in lowered
        or (hostname is not None and hostname.endswith(".invalid"))
    )


def _arn_region(value: str) -> str | None:
    parts = value.split(":", 5)
    return parts[3] if len(parts) == 6 else None


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z]", "", key.casefold())


def _normalize_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    return value


def _context_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _repository_state(repository_root: Path) -> tuple[str | None, bool | None]:
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        state_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit_result.stdout.strip() or None, bool(state_result.stdout.strip())


def _relative_display(path: Path, repository_root: Path) -> str:
    return path.resolve().relative_to(repository_root.resolve()).as_posix()


def _string_sequence(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    return ()


def _resources_of_type(resources: Mapping[str, object], resource_type: str) -> list[Mapping[str, object]]:
    return [resource for resource in resources.values() if isinstance(resource, Mapping) and resource.get("Type") == resource_type]


def _nested(value: object, *keys: str) -> object | None:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _policy_pair_present(resources: Mapping[str, object], action: object, resource: object) -> bool:
    if not isinstance(action, str) or not isinstance(resource, str):
        return False
    for policy in _resources_of_type(resources, "AWS::IAM::Policy"):
        statements = _nested(policy, "Properties", "PolicyDocument", "Statement")
        if not isinstance(statements, list):
            continue
        for statement in statements:
            if not isinstance(statement, Mapping):
                continue
            actions = statement.get("Action", [])
            resource_values = statement.get("Resource", [])
            if isinstance(actions, str):
                actions = [actions]
            if isinstance(resource_values, str):
                resource_values = [resource_values]
            if action in actions and resource in resource_values:
                return True
    return False


def _raw_prefix_notification(bucket: Mapping[str, object]) -> bool:
    resource_type = bucket.get("Type")
    if resource_type == "Custom::S3BucketNotifications":
        configurations = _nested(
            bucket,
            "Properties",
            "NotificationConfiguration",
            "LambdaFunctionConfigurations",
        )
    else:
        configurations = _nested(
            bucket,
            "Properties",
            "NotificationConfiguration",
            "LambdaConfigurations",
        )
    if not isinstance(configurations, list):
        return False
    serialized = json.dumps(configurations, sort_keys=True)
    return "s3:ObjectCreated:*" in serialized and "knowledge/raw/" in serialized


def _check(condition: bool, success: str, failure: str, checks: list[str], errors: list[str]) -> None:
    (checks if condition else errors).append(success if condition else failure)
