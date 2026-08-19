from copy import deepcopy
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import aws_cdk as cdk
from aws_cdk import assertions
import pytest

from config.clients import ProductionIndexingConfig, resolve_client_config
from data_engineering_assistant_cdk.data_engineering_assistant_cdk_stack import (
    DataEngineeringAssistantCdkStack,
)
from knowledge.indexing_configuration import (
    AutomaticIndexingConfig,
    IndexingRuntimeMode,
    build_automatic_indexing_workflow,
    check_indexing_readiness,
)
from knowledge.indexing_errors import (
    IndexingConfigurationError,
    IndexingSecretError,
    IndexingSecretSchemaError,
    ManifestWriteConflictError,
    RedriveSafetyError,
)
from knowledge.indexing_redrive import (
    IndexingRedriveService,
    RedriveFilters,
)
from knowledge.indexing_secrets import (
    QdrantCredentials,
    SecretsManagerQdrantCredentialResolver,
)
from knowledge.manifest import KnowledgeManifestRepository
from knowledge.models import DocumentMetadata, KnowledgeManifestEntry
from knowledge.storage import (
    ConditionalStorageConflictError,
    KnowledgeKeys,
    VersionedJsonObject,
)


class MemoryStorage:
    def __init__(self, conflicts: int = 0) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.version = 0
        self.conflicts = conflicts

    def put_bytes(self, key, content, *, content_type, metadata=None):
        raise NotImplementedError

    def put_json(self, key: str, payload: Mapping[str, Any]) -> None:
        self.objects[key] = deepcopy(dict(payload))
        self.version += 1

    def get_json(self, key: str) -> dict[str, Any] | None:
        return deepcopy(self.objects.get(key))

    def get_json_versioned(self, key: str) -> VersionedJsonObject:
        return VersionedJsonObject(self.get_json(key), str(self.version) if key in self.objects else None)

    def put_json_if_version(self, key, payload, expected_version):
        if self.conflicts:
            self.conflicts -= 1
            current = self.objects.setdefault(
                key, {"schema_version": 1, "documents": {}}
            )
            current["documents"]["concurrent"] = {"preserved": True}
            self.version += 1
            raise ConditionalStorageConflictError("conflict")
        actual = str(self.version) if key in self.objects else None
        if actual != expected_version:
            raise ConditionalStorageConflictError("conflict")
        self.put_json(key, payload)


def entry(document_id: str = "doc") -> KnowledgeManifestEntry:
    return KnowledgeManifestEntry(
        document_id=document_id,
        metadata=DocumentMetadata(
            filename="safe.txt", file_type="txt", upload_timestamp="now",
            checksum="abc", source="upload", document_size=4,
        ),
        chunk_count=1, embedding_status="pending", ingestion_timestamp="now",
        raw_key=f"knowledge/raw/{document_id}/safe.txt", processed_key=None,
        chunks_key=f"knowledge/chunks/{document_id}.json",
        embedding_key=KnowledgeKeys.embeddings(document_id), pending_chunk_count=1,
    )


def production_environment(**overrides: str) -> dict[str, str]:
    values = {
        "KNOWLEDGE_AUTOMATIC_INDEXING_ENABLED": "true",
        "KNOWLEDGE_INDEXING_RUNTIME_MODE": "production",
        "KNOWLEDGE_EMBEDDING_PROVIDER": "bedrock",
        "KNOWLEDGE_VECTOR_STORE_PROVIDER": "qdrant",
        "KNOWLEDGE_QDRANT_URL": "https://vectors.example.invalid",
        "KNOWLEDGE_QDRANT_SECRET_IDENTIFIER": "secret-arn",
        "KNOWLEDGE_QDRANT_TLS_REQUIRED": "true",
        "KNOWLEDGE_QDRANT_AUTHENTICATION_REQUIRED": "true",
        "CLIENT_ID": "client-a", "DEPLOYMENT_ENVIRONMENT": "prod",
    }
    values.update(overrides)
    return values


def test_production_config_fails_closed_and_local_loopback_is_compatible():
    assert AutomaticIndexingConfig.from_environment({}).enabled is False
    with pytest.raises(IndexingConfigurationError):
        AutomaticIndexingConfig.from_environment(production_environment(KNOWLEDGE_QDRANT_URL="http://private.example"))
    local = AutomaticIndexingConfig(
        enabled=True, embedding_provider="fake", vector_store="qdrant",
        qdrant_url="http://localhost:6333", qdrant_api_key="local",
        runtime_mode=IndexingRuntimeMode.LOCAL,
    )
    assert local.enabled
    with pytest.raises(IndexingConfigurationError):
        AutomaticIndexingConfig.from_environment(
            production_environment(KNOWLEDGE_QDRANT_SECRET_IDENTIFIER="")
        )


def test_secret_is_lazy_cached_validated_and_never_exposed(caplog):
    class Client:
        calls = 0
        def get_secret_value(self, **kwargs):
            self.calls += 1
            return {"SecretString": json.dumps({"api_key": "super-secret"})}
    client = Client()
    resolver = SecretsManagerQdrantCredentialResolver("secret-arn", secrets_client=client)
    assert client.calls == 0
    assert resolver.resolve().api_key == "super-secret"
    assert resolver.resolve().api_key == "super-secret"
    assert client.calls == 1
    assert "super-secret" not in caplog.text
    with pytest.raises(IndexingSecretSchemaError) as error:
        SecretsManagerQdrantCredentialResolver._parse_response(
            {"SecretString": '{"api_key":"value","unexpected":true}'}
        )
    assert "value" not in str(error.value)
    class Failed:
        def get_secret_value(self, **kwargs):
            raise RuntimeError("super-secret")
    with pytest.raises(IndexingSecretError) as error:
        SecretsManagerQdrantCredentialResolver("id", secrets_client=Failed()).resolve()
    assert "super-secret" not in str(error.value)


def test_readiness_is_configuration_only():
    report = check_indexing_readiness(
        AutomaticIndexingConfig.from_environment(production_environment())
    )
    assert report.ready and "tls_required" in report.checks


def test_production_composition_is_lazy_and_authenticated_without_connection():
    class Resolver:
        calls = 0
        def resolve(self):
            self.calls += 1
            return QdrantCredentials("secret")
    resolver = Resolver()
    factory_calls = []
    storage = MemoryStorage()
    manifest = KnowledgeManifestRepository(storage)
    workflow = build_automatic_indexing_workflow(
        AutomaticIndexingConfig.from_environment(production_environment()),
        storage=storage,
        manifest=manifest,
        credential_resolver=resolver,
        qdrant_store_factory=lambda **kwargs: factory_calls.append(kwargs),
    )
    assert workflow is not None
    assert resolver.calls == 0
    assert factory_calls == []


def test_manifest_reconciles_conflict_and_preserves_concurrent_entry():
    storage = MemoryStorage(conflicts=1)
    KnowledgeManifestRepository(storage, maximum_conflict_retries=2).upsert(entry())
    documents = storage.objects[KnowledgeKeys.MANIFEST]["documents"]
    assert set(documents) == {"concurrent", "doc"}


def test_manifest_conflict_exhaustion_is_typed():
    storage = MemoryStorage(conflicts=5)
    with pytest.raises(ManifestWriteConflictError):
        KnowledgeManifestRepository(storage, maximum_conflict_retries=1).upsert(entry())


def descriptor(client="client-a", failure="TimeoutError"):
    return {
        "document_id": "doc", "client_id": client, "environment": "prod",
        "namespace": "ns", "domain": "domain", "index_status": "failed",
        "chunks": [
            {"chunk_id": "one", "status": "indexed", "last_error_type": None},
            {"chunk_id": "two", "status": "pending", "last_error_type": failure},
        ],
    }


def redrive_storage(client="client-a", failure="TimeoutError"):
    storage = MemoryStorage()
    item = entry().to_dict()
    storage.put_json(KnowledgeKeys.MANIFEST, {"schema_version": 1, "documents": {"doc": item}})
    storage.put_json(item["embedding_key"], descriptor(client, failure))
    return storage


def test_redrive_is_dry_by_default_and_apply_is_explicit():
    storage = redrive_storage()
    service = IndexingRedriveService(storage)
    filters = RedriveFilters("client-a", "prod")
    dry = service.redrive(filters)
    assert dry.dry_run and dry.descriptors_updated == 0
    dispatched = []
    applied = service.redrive(filters, apply=True, reset_retryable=True, dispatcher=dispatched.append)
    assert applied.descriptors_updated == 1 and applied.documents_dispatched == 1
    assert storage.get_json(KnowledgeKeys.embeddings("doc"))["chunks"][1]["last_error_type"] is None


def test_redrive_rejects_cross_client_and_permanent_failure():
    with pytest.raises(RedriveSafetyError):
        IndexingRedriveService(redrive_storage("client-b")).inspect(
            RedriveFilters("client-a", "prod")
        )
    dispatched = []
    report = IndexingRedriveService(redrive_storage(failure="ValueError")).redrive(
        RedriveFilters("client-a", "prod"), apply=True,
        reset_retryable=True, dispatcher=dispatched.append,
    )
    assert not dispatched and report.documents_dispatched == 0


def enabled_infrastructure(
    vpc=False,
    reserved_concurrent_executions=None,
) -> ProductionIndexingConfig:
    return ProductionIndexingConfig(
        enabled=True, bedrock_model_arn="arn:aws:bedrock:us-west-2::foundation-model/model",
        embedding_dimensions=1024,
        qdrant_url="https://vectors.example.invalid", qdrant_collection="client_a_prod",
        qdrant_secret_arn="arn:aws:secretsmanager:us-west-2:111122223333:secret:qdrant",
        dependency_layer_arn="arn:aws:lambda:us-west-2:111122223333:layer:qdrant:1",
        vpc_id="vpc-123" if vpc else None,
        subnet_ids=("subnet-1", "subnet-2") if vpc else (),
        availability_zones=("us-west-2a", "us-west-2b") if vpc else (),
        qdrant_security_group_id="sg-123" if vpc else None,
        reserved_concurrent_executions=reserved_concurrent_executions,
    )


def test_cdk_default_has_no_production_iam_layer_or_network():
    app = cdk.App()
    template = assertions.Template.from_stack(DataEngineeringAssistantCdkStack(app, "default"))
    assert not template.find_resources("AWS::EC2::SecurityGroup")
    function = template.find_resources("AWS::Lambda::Function")["DocumentIngestionFunction"]
    assert "Layers" not in function["Properties"]
    policies = json.dumps(template.find_resources("AWS::IAM::Policy"))
    assert "secretsmanager:GetSecretValue" not in policies
    ingestion_policy = next(
        value for key, value in template.find_resources("AWS::IAM::Policy").items()
        if key.startswith("DocumentIngestionRoleDefaultPolicy")
    )
    assert "bedrock:InvokeModel" not in json.dumps(ingestion_policy)


def test_cdk_enabled_has_scoped_iam_layer_and_optional_network():
    app = cdk.App()
    stack = DataEngineeringAssistantCdkStack(
        app, "enabled", client_id="client-a", environment="prod",
        production_indexing=enabled_infrastructure(vpc=True)
    )
    template = assertions.Template.from_stack(stack)
    function = template.find_resources("AWS::Lambda::Function")["DocumentIngestionFunction"]
    assert len(function["Properties"]["Layers"]) == 1
    policies = json.dumps(template.find_resources("AWS::IAM::Policy"))
    assert "secretsmanager:GetSecretValue" in policies
    assert "bedrock:InvokeModel" in policies
    assert "arn:aws:bedrock:us-west-2::foundation-model/model" in policies
    assert "knowledge/chunks/*" in policies
    assert template.find_resources("AWS::EC2::SecurityGroup")
    assert template.find_resources("AWS::EC2::VPCEndpoint")


def test_cdk_supported_environment_can_opt_in_to_reserved_concurrency():
    app = cdk.App()
    stack = DataEngineeringAssistantCdkStack(
        app,
        "enabled-with-reserved-concurrency",
        client_id="client-a",
        environment="prod",
        production_indexing=enabled_infrastructure(
            reserved_concurrent_executions=5
        ),
    )
    template = assertions.Template.from_stack(stack)
    properties = template.find_resources("AWS::Lambda::Function")[
        "DocumentIngestionFunction"
    ]["Properties"]

    assert properties["ReservedConcurrentExecutions"] == 5
    assert properties["Runtime"] == "python3.12"
    assert properties["Timeout"] == 300
    assert properties["MemorySize"] == 512
    assert len(properties["Layers"]) == 1
    assert "DeadLetterConfig" in properties


def test_production_indexing_rejects_zero_reserved_concurrency():
    with pytest.raises(ValueError, match="limits are invalid"):
        enabled_infrastructure(reserved_concurrent_executions=0)


def test_context_rejects_plaintext_secret_and_packaging_is_pinned():
    with pytest.raises(ValueError):
        resolve_client_config({"indexingQdrantApiKey": "secret"})
    requirements = open("lambda/indexing_runtime_requirements.txt", encoding="utf-8").read()
    assert "qdrant-client==" in requirements


def test_indexing_runtime_layer_uses_a_fully_pinned_linux_lock():
    direct_requirements = Path(
        "lambda/indexing_runtime_requirements.txt"
    ).read_text(encoding="utf-8")
    lock_lines = [
        line.strip()
        for line in Path(
            "lambda/indexing_runtime_requirements.lock.txt"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    build_script = Path(
        "scripts/build_indexing_runtime_layer.ps1"
    ).read_text(encoding="utf-8")

    assert "qdrant-client==1.18.0" in direct_requirements
    assert "qdrant-client==1.18.0" in lock_lines
    assert len(lock_lines) == len(set(lock_lines))
    assert all("==" in requirement for requirement in lock_lines)
    assert "indexing_runtime_requirements.lock.txt" in build_script
    assert "--no-deps" in build_script
    assert "--platform manylinux2014_x86_64" in build_script
    assert "--python-version 3.12" in build_script
    assert "--abi cp312" in build_script
