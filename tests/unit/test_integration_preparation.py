from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import subprocess

import aws_cdk as cdk
from aws_cdk import assertions
import pytest

from config.clients import resolve_client_config
from data_engineering_assistant_cdk.data_engineering_assistant_cdk_stack import (
    DataEngineeringAssistantCdkStack,
)
from knowledge.integration_preparation import (
    FieldStatus,
    bootstrap_local_configuration,
    build_audit_artifact,
    configuration_fingerprint,
    expected_resource_report,
    generate_context_artifacts,
    require_generated_output_path,
    require_repository_path,
    review_configuration,
    review_synthesized_template,
    write_expected_resource_report,
)
from knowledge.integration_validation import IntegrationPreflightValidator
from scripts.prepare_indexing_integration import main as preparation_main


ROOT = Path(__file__).parents[2]
EXAMPLE = ROOT / "config" / "integration-validation.internal-dev.example.json"


def valid_context(*, vpc: bool = False) -> dict[str, object]:
    context = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    context.update(
        {
            "indexingQdrantUrl": "https://qdrant.internal.example.com",
            "indexingQdrantSecretArn": (
                "arn:aws:secretsmanager:us-west-2:123456789012:"
                "secret:internal-dev-qdrant"
            ),
            "indexingDependencyLayerArn": (
                "arn:aws:lambda:us-west-2:123456789012:"
                "layer:internal-dev-qdrant:1"
            ),
        }
    )
    if vpc:
        context.update(
            {
                "indexingVpcId": "vpc-123abc",
                "indexingSubnetIds": ["subnet-123abc", "subnet-456def"],
                "indexingAvailabilityZones": ["us-west-2a", "us-west-2b"],
                "indexingQdrantSecurityGroupId": "sg-123abc",
            }
        )
    return context


def fake_repository(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / ".local").mkdir()
    (tmp_path / "build").mkdir()
    (tmp_path / ".gitignore").write_text(
        "config/*.local.json\n.local/\nbuild/\n", encoding="utf-8"
    )
    return tmp_path


def test_local_configuration_creation_preserves_placeholders(tmp_path):
    root = fake_repository(tmp_path)
    source = root / "config" / "integration-validation.internal-dev.example.json"
    source.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    output = root / "config" / "integration-validation.internal-dev.local.json"

    created = bootstrap_local_configuration(
        source=source, output=output, repository_root=root
    )

    assert created == output
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(
        source.read_text(encoding="utf-8")
    )
    assert "000000000000" in output.read_text(encoding="utf-8")


def test_bootstrap_refuses_overwrite_and_force_is_bounded(tmp_path):
    root = fake_repository(tmp_path)
    source = root / "config" / "integration-validation.internal-dev.example.json"
    source.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    output = root / "config" / "integration-validation.internal-dev.local.json"
    output.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        bootstrap_local_configuration(
            source=source, output=output, repository_root=root
        )
    bootstrap_local_configuration(
        source=source, output=output, force=True, repository_root=root
    )
    assert "internal-dev" in output.read_text(encoding="utf-8")


def test_repository_bound_path_validation_and_output_roots(tmp_path):
    root = fake_repository(tmp_path / "repository")
    outside = tmp_path / "outside.json"
    with pytest.raises(ValueError):
        require_repository_path(outside, repository_root=root)
    with pytest.raises(ValueError):
        require_generated_output_path(root / "config" / "artifact.json", repository_root=root)
    assert require_generated_output_path(root / ".local" / "artifact.json", repository_root=root).is_relative_to(root)


def test_default_generated_paths_are_git_ignored():
    for path in (
        "config/integration-validation.internal-dev.local.json",
        ".local/internal-dev-indexing-context.ps1",
        ".local/internal-dev-indexing-audit.json",
    ):
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", path],
            cwd=ROOT,
            check=False,
        )
        assert result.returncode == 0


def statuses(report) -> dict[str, FieldStatus]:
    return {field.field: field.status for field in report.fields}


def test_example_review_detects_placeholders_and_reports_every_category():
    report = review_configuration(
        json.loads(EXAMPLE.read_text(encoding="utf-8")),
        preflight=IntegrationPreflightValidator(python_version=lambda: (3, 12)),
    )
    result = statuses(report)
    assert not report.ready
    assert result["indexingQdrantUrl"] is FieldStatus.PLACEHOLDER
    assert result["indexingQdrantSecretArn"] is FieldStatus.PLACEHOLDER
    assert result["indexingDependencyLayerArn"] is FieldStatus.PLACEHOLDER
    assert result["indexingReservedConcurrentExecutions"] is (
        FieldStatus.OPTIONAL_NOT_CONFIGURED
    )
    assert result["indexingVpcId"] is FieldStatus.OPTIONAL_NOT_CONFIGURED
    assert result["knowledgeBucketReference"] is FieldStatus.READY


def test_review_detects_missing_invalid_arn_and_invalid_https():
    context = valid_context()
    del context["clientId"]
    context["indexingBedrockModelArn"] = "not-an-arn"
    context["indexingQdrantUrl"] = "http://qdrant.example.com?api_key=no"
    result = statuses(review_configuration(context))
    assert result["clientId"] is FieldStatus.MISSING
    assert result["indexingBedrockModelArn"] is FieldStatus.INVALID
    assert result["indexingQdrantUrl"] is FieldStatus.INVALID


def test_review_rejects_zero_reserved_concurrency():
    context = valid_context()
    context["indexingReservedConcurrentExecutions"] = 0

    result = statuses(review_configuration(context))

    assert result["indexingReservedConcurrentExecutions"] is (
        FieldStatus.INVALID
    )


def test_plaintext_api_key_is_prohibited_and_never_printed(capsys, tmp_path):
    secret = "must-never-appear"
    path = tmp_path / "context.json"
    context = valid_context()
    context["indexingQdrantApiKey"] = secret
    path.write_text(json.dumps(context), encoding="utf-8")

    exit_code = preparation_main(["review", "--config", str(path), "--format", "json"])

    output = capsys.readouterr().out
    assert exit_code == 2
    assert secret not in output
    assert '"status": "prohibited"' in output


def test_optional_vpc_must_be_complete_and_accepts_json_arrays():
    incomplete = valid_context()
    incomplete["indexingVpcId"] = "vpc-123abc"
    report = review_configuration(incomplete)
    assert not report.ready
    assert statuses(report)["indexingSubnetIds"] is FieldStatus.INVALID

    complete = review_configuration(valid_context(vpc=True))
    assert complete.ready
    assert statuses(complete)["indexingVpcId"] is FieldStatus.READY


def test_review_command_has_human_and_json_modes(capsys, tmp_path):
    path = tmp_path / "context.json"
    path.write_text(json.dumps(valid_context()), encoding="utf-8")
    assert preparation_main(["review", "--config", str(path)]) == 0
    assert "offline configuration review: READY" in capsys.readouterr().out
    assert preparation_main(["review", "--config", str(path), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True and payload["network_calls"] == 0


def test_context_generation_normalizes_values_and_writes_redacted_audit(tmp_path):
    root = fake_repository(tmp_path)
    config_path = root / "config" / "integration-validation.internal-dev.local.json"
    context = valid_context(vpc=True)
    config_path.write_text(json.dumps(context), encoding="utf-8")
    context_path, audit_path, audit = generate_context_artifacts(
        context,
        config_path=config_path,
        context_output=root / ".local" / "context.ps1",
        audit_output=root / ".local" / "audit.json",
        repository_root=root,
        timestamp=datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc),
    )
    script = context_path.read_text(encoding="utf-8")
    assert "automaticIndexingEnabled=true" in script
    assert "indexingSubnetIds=subnet-123abc,subnet-456def" in script
    assert "SAFE ONLY FOR: cdk synth --no-lookups" in script
    assert "NOT APPROVED FOR: cdk diff or cdk deploy" in script
    serialized_audit = audit_path.read_text(encoding="utf-8")
    assert audit["configuration_fingerprint_sha256"] == configuration_fingerprint(context)
    assert audit["expected_stack_name"] == "DataEngineeringAssistantCdkStack"
    assert "must-never-appear" not in serialized_audit
    assert audit["secret_values_recorded"] is False


def test_context_generation_refuses_failed_review_unsafe_output_and_overwrite(tmp_path):
    root = fake_repository(tmp_path)
    config_path = root / "config" / "integration-validation.internal-dev.local.json"
    invalid = valid_context()
    invalid["indexingQdrantUrl"] = "https://qdrant.invalid"
    with pytest.raises(ValueError, match="pass offline review"):
        generate_context_artifacts(
            invalid,
            config_path=config_path,
            context_output=root / ".local" / "context.ps1",
            audit_output=root / ".local" / "audit.json",
            repository_root=root,
        )
    with pytest.raises(ValueError, match="under .local or build"):
        generate_context_artifacts(
            valid_context(),
            config_path=config_path,
            context_output=root / "context.ps1",
            audit_output=root / ".local" / "audit.json",
            repository_root=root,
        )
    (root / ".local" / "context.ps1").write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        generate_context_artifacts(
            valid_context(),
            config_path=config_path,
            context_output=root / ".local" / "context.ps1",
            audit_output=root / ".local" / "audit.json",
            repository_root=root,
        )


def test_non_secret_fingerprint_is_deterministic_and_excludes_plaintext_values():
    first = valid_context()
    second = dict(reversed(list(first.items())))
    assert configuration_fingerprint(first) == configuration_fingerprint(second)
    first["qdrantApiKey"] = "first-secret"
    second["qdrantApiKey"] = "second-secret"
    assert configuration_fingerprint(first) == configuration_fingerprint(second)


def test_audit_artifact_redacts_prohibited_fields(tmp_path):
    root = fake_repository(tmp_path)
    context = valid_context()
    context["knowledgeQdrantApiKey"] = "never-record-me"
    preflight = IntegrationPreflightValidator(python_version=lambda: (3, 12)).validate(valid_context())
    artifact = build_audit_artifact(
        context,
        config_path=root / "config" / "input.local.json",
        context_path=root / ".local" / "context.ps1",
        preflight=preflight,
        repository_root=root,
        timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    serialized = json.dumps(artifact)
    assert "never-record-me" not in serialized
    assert "knowledgeQdrantApiKey" not in serialized


def synth_template(context: dict[str, object]) -> dict[str, object]:
    resolved = resolve_client_config(context)
    app = cdk.App()
    stack = DataEngineeringAssistantCdkStack(
        app,
        "integration-preparation-test",
        client_id=resolved.client_id,
        environment=resolved.environment,
        production_indexing=resolved.production_indexing,
    )
    return assertions.Template.from_stack(stack).to_json()


def test_expected_resource_report_asserts_enabled_template_and_no_network(monkeypatch):
    def forbidden_network(*args, **kwargs):
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", forbidden_network)
    context = valid_context(vpc=True)
    report = expected_resource_report(context, template=synth_template(context))
    assert report["template_review"]["valid"] is True
    assert report["network_calls"] == 0
    assert "reserved_concurrency_matches_configuration" in (
        report["template_review"]["checks"]
    )
    assert "qdrant_infrastructure_absent" in report["template_review"]["checks"]
    assert "production_client_resources_absent" in report["template_review"]["checks"]


def test_template_review_rejects_production_client_leakage():
    context = valid_context()
    template = synth_template(context)
    function = template["Resources"]["DocumentIngestionFunction"]
    function["Properties"]["Environment"]["Variables"]["DEPLOYMENT_ENVIRONMENT"] = "prod"
    report = review_synthesized_template(template, context)
    assert not report.valid
    assert "indexing_environment_invalid" in report.errors
    assert "production_client_resource_leakage" in report.errors


def test_expected_report_output_refuses_overwrite_and_outside_paths(tmp_path):
    root = fake_repository(tmp_path)
    output = root / ".local" / "expected.json"
    write_expected_resource_report(valid_context(), output=output, repository_root=root)
    with pytest.raises(FileExistsError):
        write_expected_resource_report(valid_context(), output=output, repository_root=root)
    with pytest.raises(ValueError):
        write_expected_resource_report(
            valid_context(), output=root / "expected.json", repository_root=root
        )
