from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import socket
from zipfile import ZipFile

import pytest

from knowledge.chunking import TextChunker
from knowledge.indexing_errors import IntegrationValidationError
from knowledge.indexing_secrets import SecretsManagerQdrantCredentialResolver
from knowledge.integration_validation import (
    IntegrationEvidence,
    IntegrationEvidenceExpectation,
    IntegrationEvidenceVerifier,
    IntegrationFailureDisposition,
    IntegrationPhase,
    IntegrationPreflightValidator,
    IntegrationValidationRunner,
    classify_integration_failure,
)
from scripts.inspect_indexing_runtime_layer import inspect_layer_archive
from scripts.validate_indexing_integration import main as preflight_main


ROOT = Path(__file__).parents[2]
CONFIG_PATH = (
    ROOT / "config" / "integration-validation.internal-dev.example.json"
)
FIXTURE_PATH = ROOT / "data" / "integration" / "vector-indexing-validation.txt"
EXPECTATIONS_PATH = FIXTURE_PATH.with_suffix(".expectations.json")


def valid_context() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_offline_preflight_succeeds_without_network(monkeypatch):
    def forbidden_network(*args, **kwargs):
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", forbidden_network)
    report = IntegrationPreflightValidator(
        python_version=lambda: (3, 12)
    ).validate(valid_context())

    assert report.ready
    assert report.client_id == "internal"
    assert report.environment == "dev"
    assert report.to_dict()["network_calls"] == 0


@pytest.mark.parametrize(
    ("key", "value", "expected_error"),
    [
        ("indexingBedrockModelArn", "not-an-arn", "bedrock_arn_invalid"),
        ("indexingQdrantUrl", "http://remote.invalid", "existing_indexing_configuration_invalid"),
        ("indexingQdrantSecretArn", "secret-name", "secret_arn_invalid"),
        ("indexingDependencyLayerArn", "layer-name", "layer_arn_invalid"),
        ("indexingQdrantCollection", "shared", "collection_scope_invalid"),
        ("indexingEmbeddingDimensions", 0, "existing_indexing_configuration_invalid"),
        ("indexingRequestTimeoutSeconds", 301, "timeout_or_retry_bounds_invalid"),
        ("indexingMaximumChunksPerInvocation", 10001, "work_limits_invalid"),
        ("indexingQdrantKmsKeyArn", "not-an-arn", "kms_key_arn_invalid"),
    ],
)
def test_offline_preflight_rejects_invalid_configuration(
    key, value, expected_error
):
    context = valid_context()
    context[key] = value
    report = IntegrationPreflightValidator(
        python_version=lambda: (3, 12)
    ).validate(context)
    assert not report.ready
    assert expected_error in report.errors or expected_error in report.checks


def test_preflight_rejects_plaintext_api_key_without_exposing_value():
    context = valid_context()
    context["indexingQdrantApiKey"] = "credential-value-that-must-not-appear"
    report = IntegrationPreflightValidator(
        python_version=lambda: (3, 12)
    ).validate(context)
    output = json.dumps(report.to_dict())
    assert not report.ready
    assert "plaintext_api_key_forbidden" in report.errors
    assert "credential-value-that-must-not-appear" not in output


def test_preflight_command_returns_nonzero_and_safe_output(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "invalid.json"
    context = valid_context()
    context["indexingQdrantApiKey"] = "never-print-this-value"
    path.write_text(json.dumps(context), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["validate_indexing_integration", "--config", str(path)],
    )
    assert preflight_main() == 2
    output = capsys.readouterr().out
    assert "plaintext_api_key_forbidden" in output
    assert "never-print-this-value" not in output


@pytest.mark.parametrize(
    ("client", "client_id", "environment"),
    [
        ("demo-client-dev", "demo-client", "dev"),
        ("internal-dev", "internal", "prod"),
    ],
)
def test_integration_gate_rejects_demo_and_production_scope(
    client, client_id, environment
):
    context = valid_context()
    context.update(
        {"client": client, "clientId": client_id, "environment": environment}
    )
    report = IntegrationPreflightValidator(
        python_version=lambda: (3, 12)
    ).validate(context)
    assert not report.ready


def test_preflight_rejects_incomplete_vpc_configuration():
    context = valid_context()
    context["indexingVpcId"] = "vpc-abc123"
    report = IntegrationPreflightValidator(
        python_version=lambda: (3, 12)
    ).validate(context)
    assert not report.ready
    assert "existing_indexing_configuration_invalid" in report.errors


def test_fixture_is_synthetic_licensed_and_deterministically_chunked():
    text = FIXTURE_PATH.read_text(encoding="utf-8")
    expected = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    chunks = TextChunker(
        expected["chunk_size"], expected["chunk_overlap"]
    ).chunk("integration-fixture", text)
    assert expected["data_classification"] == "repository-authored-synthetic"
    assert expected["license"] == "MIT"
    assert len(chunks) == expected["expected_chunk_count"]
    assert all(text.count(phrase) == 1 for phrase in expected["retrieval_phrases"])
    assert expected["expected_client"] == "internal"
    assert expected["expected_environment"] == "dev"


def test_phase_plan_is_dry_run_and_marks_cost_and_mutation():
    plan = IntegrationValidationRunner(
        preflight=IntegrationPreflightValidator(
            python_version=lambda: (3, 12)
        )
    ).plan(valid_context())
    assert plan.dry_run
    assert tuple(operation.phase for operation in plan.operations) == tuple(
        IntegrationPhase
    )
    cleanup = plan.operations[-1]
    assert cleanup.phase == IntegrationPhase.CLEANUP_PLAN
    assert cleanup.manual_only and cleanup.state_changing and cleanup.billable


def test_state_change_requires_confirmation_and_injected_executor():
    runner = IntegrationValidationRunner(
        preflight=IntegrationPreflightValidator(
            python_version=lambda: (3, 12)
        )
    )
    assert runner.execute(valid_context()) == ()
    with pytest.raises(IntegrationValidationError):
        runner.execute(valid_context(), apply=True)
    with pytest.raises(IntegrationValidationError):
        runner.execute(
            valid_context(),
            apply=True,
            confirmation=runner.CONFIRMATION,
        )


def test_injected_executor_receives_scope_and_safe_results_only():
    class Executor:
        def __init__(self):
            self.calls = []

        def execute(self, operation, *, client_id, environment):
            self.calls.append((operation.phase, client_id, environment))
            return {"document_id": "fixture", "observed_count": 1}

    executor = Executor()
    runner = IntegrationValidationRunner(
        preflight=IntegrationPreflightValidator(
            python_version=lambda: (3, 12)
        ),
        executor=executor,
    )
    results = runner.execute(
        valid_context(),
        apply=True,
        confirmation=runner.CONFIRMATION,
        phases=(IntegrationPhase.UPLOAD_PLAN,),
    )
    assert results == ({"document_id": "fixture", "observed_count": 1},)
    assert executor.calls == [
        (IntegrationPhase.UPLOAD_PLAN, "internal", "dev")
    ]


def test_layer_archive_inspection_reports_structure_version_native_files_and_hash(tmp_path):
    archive_path = tmp_path / "layer.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("python/qdrant_client/__init__.py", "")
        archive.writestr(
            "python/qdrant_client-1.18.0.dist-info/METADATA",
            "Name: qdrant-client\nVersion: 1.18.0\n",
        )
        archive.writestr("python/grpc/_cython/cygrpc.so", b"synthetic")
    report = inspect_layer_archive(
        archive_path, expected_qdrant_version="1.18.0"
    )
    assert report.valid
    assert report.python_root_present
    assert report.qdrant_client_version == "1.18.0"
    assert report.native_file_count == 1
    assert report.native_suffixes == (".so",)
    assert len(report.sha256) == 64 and report.archive_size_bytes > 0

    mismatch = inspect_layer_archive(
        archive_path, expected_qdrant_version="9.9.9"
    )
    assert not mismatch.valid
    assert "qdrant_client_version_mismatch" in mismatch.errors


@pytest.mark.parametrize(
    ("error_type", "disposition"),
    [
        ("IndexingSecretSchemaError", IntegrationFailureDisposition.PERMANENT),
        ("AuthenticationError", IntegrationFailureDisposition.OPERATOR_ACTION),
        ("TlsError", IntegrationFailureDisposition.OPERATOR_ACTION),
        ("VectorStoreUnavailableError", IntegrationFailureDisposition.RETRYABLE),
        ("EmbeddingAccessDeniedError", IntegrationFailureDisposition.OPERATOR_ACTION),
        ("EmbeddingThrottledError", IntegrationFailureDisposition.RETRYABLE),
        ("VectorDimensionMismatchError", IntegrationFailureDisposition.PERMANENT),
        ("VectorCollectionConfigurationError", IntegrationFailureDisposition.PERMANENT),
        ("ManifestWriteConflictError", IntegrationFailureDisposition.RETRYABLE),
        ("ConditionalStorageConflictError", IntegrationFailureDisposition.RETRYABLE),
        ("AutomaticIndexingIncompleteError", IntegrationFailureDisposition.RETRYABLE),
        ("PartialIndexingRetry", IntegrationFailureDisposition.RETRYABLE),
        ("RetriesExhaustedDlqDelivery", IntegrationFailureDisposition.OPERATOR_ACTION),
        ("PermanentFailureResetAttempt", IntegrationFailureDisposition.PERMANENT),
        ("CrossClientRedriveAttempt", IntegrationFailureDisposition.PERMANENT),
        ("SecretRotationRequired", IntegrationFailureDisposition.OPERATOR_ACTION),
    ],
)
def test_failure_path_classification(error_type, disposition):
    assert classify_integration_failure(error_type) == disposition


def test_secret_rotation_requires_a_new_resolver_execution_environment():
    class Client:
        def __init__(self, value):
            self.value = value

        def get_secret_value(self, **kwargs):
            return {"SecretString": json.dumps({"api_key": self.value})}

    old_environment = SecretsManagerQdrantCredentialResolver(
        "secret-arn", secrets_client=Client("old-value")
    )
    new_environment = SecretsManagerQdrantCredentialResolver(
        "secret-arn", secrets_client=Client("new-value")
    )
    assert old_environment.resolve().api_key == "old-value"
    assert old_environment.resolve().api_key == "old-value"
    assert new_environment.resolve().api_key == "new-value"


def test_offline_evidence_verifier_checks_scope_state_and_safe_logs():
    expectation = IntegrationEvidenceExpectation(
        client_id="internal",
        environment="dev",
        namespace="data-engineering-integration",
        domain="integration-validation",
        collection="internal_dev_integration",
        embedding_dimensions=1024,
        expected_chunk_count=1,
        forbidden_log_phrases=("fixture body",),
    )
    descriptor = {
        "schema_version": 2,
        "client_id": "internal",
        "environment": "dev",
        "namespace": expectation.namespace,
        "domain": expectation.domain,
        "index_status": "complete",
        "vector_dimension": 1024,
        "vector_collection": expectation.collection,
        "chunks": [{"chunk_id": "one", "status": "indexed"}],
    }
    manifest = {
        "index_status": "complete",
        "vector_dimension": 1024,
        "vector_collection": expectation.collection,
    }
    payload = {
        "client_id": "internal",
        "environment": "dev",
        "namespace": expectation.namespace,
        "domain": expectation.domain,
        "collection": expectation.collection,
        "dimensions": 1024,
    }
    report = IntegrationEvidenceVerifier().verify(
        IntegrationEvidence(
            s3_event_delivered=True,
            descriptor=descriptor,
            manifest_entry=manifest,
            vector_payload_summaries=(payload,),
            touched_scopes=(("internal", "dev"),),
            touched_collections=(expectation.collection,),
            touched_prefixes=("knowledge/raw/integration/fixture.txt",),
            dlq_message_count=0,
            permanent_failure_dispatch_count=0,
            log_records=(
                '{"client_id":"internal","document_id":"fixture"}',
            ),
        ),
        expectation,
    )
    assert report.valid

    cross_client = deepcopy(payload)
    cross_client["client_id"] = "another-client"
    invalid = IntegrationEvidenceVerifier().verify(
        IntegrationEvidence(
            s3_event_delivered=True,
            descriptor=descriptor,
            manifest_entry=manifest,
            vector_payload_summaries=(cross_client,),
            touched_scopes=(("internal", "dev"), ("another-client", "dev")),
            touched_collections=(
                expectation.collection,
                "another-client-dev",
            ),
            touched_prefixes=(
                "knowledge/raw/integration/fixture.txt",
                "knowledge/raw/another-client/fixture.txt",
            ),
            dlq_message_count=0,
            permanent_failure_dispatch_count=0,
            log_records=(),
        ),
        expectation,
    )
    assert not invalid.valid
    assert "cross_scope_touch_detected" in invalid.errors

    partial_descriptor = deepcopy(descriptor)
    partial_descriptor.update(
        {
            "index_status": "partial",
            "chunks": [
                {"chunk_id": "one", "status": "indexed"},
                {
                    "chunk_id": "two",
                    "status": "pending",
                    "last_error_type": "EmbeddingThrottledError",
                },
            ],
        }
    )
    partial_manifest = deepcopy(manifest)
    partial_manifest["index_status"] = "partial"
    resumable = IntegrationEvidenceVerifier().verify(
        IntegrationEvidence(
            s3_event_delivered=True,
            descriptor=partial_descriptor,
            manifest_entry=partial_manifest,
            vector_payload_summaries=(payload,),
            touched_scopes=(("internal", "dev"),),
            touched_collections=(expectation.collection,),
            touched_prefixes=("knowledge/raw/integration/fixture.txt",),
            dlq_message_count=1,
            permanent_failure_dispatch_count=0,
            log_records=(),
        ),
        replace(
            expectation,
            expected_chunk_count=2,
            expect_complete=False,
            expect_dlq_delivery=True,
        ),
    )
    assert resumable.valid
    assert "descriptor_resumable_state_valid" in resumable.checks
    assert "dlq_evidence_valid" in resumable.checks
