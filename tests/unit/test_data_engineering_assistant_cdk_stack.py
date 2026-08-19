from pathlib import Path
import re
import subprocess
import sys

import aws_cdk as core
import aws_cdk.assertions as assertions

from data_engineering_assistant_cdk.data_engineering_assistant_cdk_stack import DataEngineeringAssistantCdkStack
from data_engineering_assistant_cdk.lambda_asset import (
    DocumentIngestionAssetBundler,
)


def _template() -> assertions.Template:
    app = core.App()
    stack = DataEngineeringAssistantCdkStack(app, "data-engineering-assistant-cdk")
    return assertions.Template.from_stack(stack)


def _stack_and_template(
    client_id: str = "internal",
    environment: str = "dev",
) -> tuple[DataEngineeringAssistantCdkStack, assertions.Template]:
    app = core.App()
    stack = DataEngineeringAssistantCdkStack(
        app,
        f"{client_id}-{environment}",
        client_id=client_id,
        environment=environment,
    )
    return stack, assertions.Template.from_stack(stack)


def test_client_and_environment_tags_are_applied_to_the_stack():
    template = _template()

    for resource in template.find_resources("AWS::S3::Bucket").values():
        tags = resource["Properties"]["Tags"]
        assert {"Key": "ClientId", "Value": "internal"} in tags
        assert {"Key": "Environment", "Value": "dev"} in tags


def test_client_context_outputs_are_created():
    template = _template()

    template.has_output("ClientId", {"Value": "internal"})
    template.has_output("DeploymentEnvironment", {"Value": "dev"})
    template.has_output(
        "ResourcePrefix",
        {"Value": "bah-de-assistant-internal-dev"},
    )


def test_demo_client_dev_synthesizes_client_specific_resource_names():
    internal_stack, internal_template = _stack_and_template()
    demo_stack, demo_template = _stack_and_template(client_id="demo-client")

    assert internal_stack.stack_name != demo_stack.stack_name
    assert all(
        "BucketName" not in resource.get("Properties", {})
        for resource in internal_template.find_resources(
            "AWS::S3::Bucket"
        ).values()
    )

    demo_buckets = demo_template.find_resources("AWS::S3::Bucket").values()
    demo_bucket_names = [
        str(resource["Properties"]["BucketName"])
        for resource in demo_buckets
        if "BucketName" in resource["Properties"]
    ]
    assert len(demo_bucket_names) == 6
    assert all(
        "bah-de-assistant-demo-client-dev" in name
        for name in demo_bucket_names
    )


def test_six_secure_s3_buckets_are_created():
    template = _template()

    template.resource_count_is("AWS::S3::Bucket", 6)
    template.resource_count_is("AWS::S3::BucketPolicy", 6)
    template.all_resources_properties(
        "AWS::S3::Bucket",
        {
            "BucketEncryption": {
                "ServerSideEncryptionConfiguration": [
                    {
                        "ServerSideEncryptionByDefault": {
                            "SSEAlgorithm": "AES256",
                        }
                    }
                ]
            },
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "BlockPublicPolicy": True,
                "IgnorePublicAcls": True,
                "RestrictPublicBuckets": True,
            },
        },
    )

    # Each generated bucket policy denies every S3 action over insecure HTTP.
    for policy in template.find_resources("AWS::S3::BucketPolicy").values():
        statements = policy["Properties"]["PolicyDocument"]["Statement"]
        assert any(
            statement.get("Effect") == "Deny"
            and statement.get("Action") == "s3:*"
            and statement.get("Condition") == {
                "Bool": {"aws:SecureTransport": "false"}
            }
            for statement in statements
        )


def test_required_versioning_and_lifecycle_rules_are_configured():
    template = _template()

    # Four durable buckets enable versioning.
    template.resource_properties_count_is(
        "AWS::S3::Bucket",
        {"VersioningConfiguration": {"Status": "Enabled"}},
        4,
    )

    # Raw objects transition to Standard-IA after 90 days.
    template.has_resource_properties(
        "AWS::S3::Bucket",
        {
            "LifecycleConfiguration": {
                "Rules": [
                    {
                        "Id": "TransitionToInfrequentAccess",
                        "Status": "Enabled",
                        "Transitions": [
                            {
                                "StorageClass": "STANDARD_IA",
                                "TransitionInDays": 90,
                            }
                        ],
                    }
                ]
            }
        },
    )

    # Development logs and Athena results expire on their required schedules.
    for rule_id, days in (("DeleteAfter90Days", 90), ("DeleteAfter30Days", 30)):
        template.has_resource_properties(
            "AWS::S3::Bucket",
            {
                "LifecycleConfiguration": {
                    "Rules": [
                        {
                            "ExpirationInDays": days,
                            "Id": rule_id,
                            "Status": "Enabled",
                        }
                    ]
                }
            },
        )


def test_required_removal_policies_and_development_cleanup_are_configured():
    template = _template()
    buckets = template.find_resources("AWS::S3::Bucket").values()

    # Durable buckets are retained; the two development buckets are deleted.
    assert sum(bucket.get("DeletionPolicy") == "Retain" for bucket in buckets) == 4
    assert sum(bucket.get("DeletionPolicy") == "Delete" for bucket in buckets) == 2

    # auto_delete_objects creates one cleanup custom resource per dev bucket.
    template.resource_count_is("Custom::S3AutoDeleteObjects", 2)


def test_all_bucket_names_are_output_without_hard_coded_names():
    template = _template()

    # CDK-generated names appear as Ref values and no bucket has BucketName set.
    for output_id in (
        "RawBucketName",
        "CuratedBucketName",
        "KnowledgeBucketName",
        "ModelsBucketName",
        "LogsBucketName",
        "AthenaResultsBucketName",
    ):
        template.has_output(output_id, {"Value": {"Ref": assertions.Match.any_value()}})

    for resource in template.find_resources("AWS::S3::Bucket").values():
        assert "BucketName" not in resource.get("Properties", {})


def test_glue_database_and_athena_workgroup_are_created():
    template = _template()

    # Preserve the requested CloudFormation logical ID.
    assert "DataEngineeringDatabase" in template.find_resources(
        "AWS::Glue::Database"
    )
    template.has_resource_properties(
        "AWS::Glue::Database",
        {
            "CatalogId": {"Ref": "AWS::AccountId"},
            "DatabaseInput": {
                "Name": "dea_catalog",
            },
        },
    )
    template.has_resource_properties(
        "AWS::Athena::WorkGroup",
        {
            "Name": "dea-workgroup",
            "State": "ENABLED",
            "WorkGroupConfiguration": {
                "EnforceWorkGroupConfiguration": True,
                "ResultConfiguration": {
                    "EncryptionConfiguration": {
                        "EncryptionOption": "SSE_S3",
                    },
                    "OutputLocation": assertions.Match.any_value(),
                },
            },
        },
    )

    # The workgroup output location resolves from the generated results bucket.
    workgroup = next(
        iter(template.find_resources("AWS::Athena::WorkGroup").values())
    )
    output_location = workgroup["Properties"]["WorkGroupConfiguration"][
        "ResultConfiguration"
    ]["OutputLocation"]
    athena_results_bucket_id = next(
        logical_id
        for logical_id in template.find_resources("AWS::S3::Bucket")
        if logical_id.startswith("AthenaResultsBucket")
    )
    assert athena_results_bucket_id in str(output_location)


def test_parameter_store_contains_all_resource_names():
    template = _template()
    parameter_names = (
        "/dea/glue/database",
        "/dea/athena/workgroup",
        "/dea/buckets/raw",
        "/dea/buckets/curated",
        "/dea/buckets/knowledge",
        "/dea/buckets/models",
        "/dea/buckets/logs",
        "/dea/buckets/athena-results",
    )

    template.resource_count_is("AWS::SSM::Parameter", 8)
    for parameter_name in parameter_names:
        template.has_resource_properties(
            "AWS::SSM::Parameter",
            {
                "Name": parameter_name,
                "Type": "String",
                "Value": assertions.Match.any_value(),
            },
        )

    # Every parameter resolves from a resource rather than a duplicated literal.
    for parameter in template.find_resources("AWS::SSM::Parameter").values():
        assert "Ref" in parameter["Properties"]["Value"]


def test_glue_database_and_athena_workgroup_names_are_output():
    template = _template()

    template.has_output(
        "GlueDatabaseName",
        {"Value": {"Ref": assertions.Match.any_value()}},
    )
    template.has_output(
        "AthenaWorkgroupName",
        {"Value": {"Ref": assertions.Match.any_value()}},
    )


def _role_and_policy(
    template: assertions.Template,
    role_prefix: str,
) -> tuple[dict, list[dict]]:
    roles = template.find_resources("AWS::IAM::Role")
    role_id, role = next(
        (logical_id, resource)
        for logical_id, resource in roles.items()
        if logical_id.startswith(role_prefix)
    )

    policies = template.find_resources("AWS::IAM::Policy")
    policy = next(
        resource
        for resource in policies.values()
        if {"Ref": role_id} in resource["Properties"]["Roles"]
    )
    return role, policy["Properties"]["PolicyDocument"]["Statement"]


def _actions(statements: list[dict]) -> set[str]:
    actions = set()
    for statement in statements:
        statement_actions = statement["Action"]
        if isinstance(statement_actions, str):
            actions.add(statement_actions)
        else:
            actions.update(statement_actions)
    return actions


def test_glue_execution_role_is_least_privilege():
    template = _template()
    role, statements = _role_and_policy(template, "GlueExecutionRole")

    assert role["Properties"]["AssumeRolePolicyDocument"]["Statement"] == [
        {
            "Action": "sts:AssumeRole",
            "Effect": "Allow",
            "Principal": {"Service": "glue.amazonaws.com"},
        }
    ]
    assert "RoleName" not in role["Properties"]

    actions = _actions(statements)
    assert {
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:GetBucketLocation",
        "s3:ListBucket",
        "s3:PutObject",
        "glue:GetDatabase",
        "glue:CreateTable",
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
    }.issubset(actions)

    # No policy statement grants access to every AWS resource.
    assert all(statement["Resource"] != "*" for statement in statements)


def test_lambda_execution_role_is_least_privilege():
    template = _template()
    role, statements = _role_and_policy(template, "LambdaExecutionRole")

    assert role["Properties"]["AssumeRolePolicyDocument"]["Statement"] == [
        {
            "Action": "sts:AssumeRole",
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
        }
    ]
    assert "RoleName" not in role["Properties"]

    actions = _actions(statements)
    assert {
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:GetBucketLocation",
        "s3:ListBucket",
        "ssm:GetParameter",
        "ssm:GetParameters",
        "bedrock:InvokeModel",
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
    }.issubset(actions)
    assert "s3:PutObject" not in actions
    assert "ssm:GetParameterHistory" not in actions
    assert "ssm:DescribeParameters" not in actions

    # No policy statement grants access to every AWS resource.
    assert all(statement["Resource"] != "*" for statement in statements)


def test_execution_role_arns_are_output():
    template = _template()

    template.has_output(
        "GlueExecutionRoleArn",
        {"Value": {"Fn::GetAtt": assertions.Match.any_value()}},
    )
    template.has_output(
        "LambdaExecutionRoleArn",
        {"Value": {"Fn::GetAtt": assertions.Match.any_value()}},
    )


def test_health_check_lambda_uses_expected_configuration_and_role():
    template = _template()
    functions = template.find_resources("AWS::Lambda::Function")

    # The L2 resource uses the exact requested CloudFormation logical ID.
    assert "HealthCheckFunction" in functions
    function = functions["HealthCheckFunction"]
    properties = function["Properties"]

    assert properties["FunctionName"] == "dea-health-check"
    assert properties["Runtime"] == "python3.12"
    assert properties["Handler"] == "index.handler"
    assert properties["Timeout"] == 30
    assert properties["MemorySize"] == 256
    assert properties["Environment"]["Variables"] == {
        "GLUE_DATABASE_PARAMETER": "/dea/glue/database"
    }

    lambda_role_id = next(
        logical_id
        for logical_id in template.find_resources("AWS::IAM::Role")
        if logical_id.startswith("LambdaExecutionRole")
    )
    assert properties["Role"] == {
        "Fn::GetAtt": [lambda_role_id, "Arn"],
    }


def test_health_check_log_group_has_expected_retention_and_cleanup():
    template = _template()
    log_groups = template.find_resources("AWS::Logs::LogGroup")
    log_group_id, log_group = next(
        (logical_id, resource)
        for logical_id, resource in log_groups.items()
        if resource["Properties"].get("LogGroupName")
        == "/aws/lambda/dea-health-check"
    )

    assert log_group["Properties"]["RetentionInDays"] == 30
    assert log_group["DeletionPolicy"] == "Delete"

    function = template.find_resources("AWS::Lambda::Function")[
        "HealthCheckFunction"
    ]
    assert function["Properties"]["LoggingConfig"]["LogGroup"] == {
        "Ref": log_group_id
    }


def test_health_check_function_name_is_output_without_api_gateway():
    template = _template()

    template.has_output(
        "HealthCheckFunctionName",
        {"Value": {"Ref": "HealthCheckFunction"}},
    )
    template.resource_count_is("AWS::ApiGateway::RestApi", 0)


def test_document_ingestion_lambda_has_bounded_runtime_configuration():
    template = _template()
    function = template.find_resources("AWS::Lambda::Function")[
        "DocumentIngestionFunction"
    ]
    properties = function["Properties"]

    assert properties["FunctionName"] == "dea-document-ingestion"
    assert properties["Runtime"] == "python3.12"
    assert properties["Handler"] == (
        "lambda.document_ingestion.index.handler"
    )
    assert properties["Timeout"] == 300
    assert properties["MemorySize"] == 512
    assert "ReservedConcurrentExecutions" not in properties
    assert "Layers" not in properties
    assert re.fullmatch(
        r"[0-9a-f]{64}\.zip",
        properties["Code"]["S3Key"],
    )
    environment = dict(properties["Environment"]["Variables"])
    bucket_reference = environment.pop("KNOWLEDGE_BUCKET_NAME")
    assert "Ref" in bucket_reference
    assert environment == {
        "CLIENT_ID": "internal",
        "DEPLOYMENT_ENVIRONMENT": "dev",
        "KNOWLEDGE_AUTOMATIC_INDEXING_ENABLED": "false",
        "KNOWLEDGE_CHUNK_OVERLAP": "100",
        "KNOWLEDGE_CHUNK_SIZE": "1000",
        "KNOWLEDGE_DOMAIN": "data-engineering",
        "KNOWLEDGE_MAXIMUM_UPLOAD_SIZE": "10485760",
        "KNOWLEDGE_NAMESPACE": "data-engineering",
        "KNOWLEDGE_RAW_PREFIX": "knowledge/raw/",
        "KNOWLEDGE_SUPPORTED_DOCUMENT_TYPES": (
            "html,json,markdown,md,pdf,py,txt"
        ),
    }
    assert "DeadLetterConfig" in properties


def test_document_ingestion_asset_contains_only_required_runtime_content(
    tmp_path: Path,
):
    bundled = DocumentIngestionAssetBundler().try_bundle(
        str(tmp_path),
        None,
    )

    assert bundled is True
    assert (tmp_path / "lambda/document_ingestion/index.py").is_file()
    assert (tmp_path / "knowledge/__init__.py").is_file()
    assert (tmp_path / "knowledge/pdf_extraction.py").is_file()
    assert (tmp_path / "pypdf/__init__.py").is_file()
    assert (tmp_path / "requests/__init__.py").is_file()
    assert (tmp_path / "certifi/cacert.pem").is_file()
    assert (tmp_path / "charset_normalizer/__init__.py").is_file()
    assert (tmp_path / "idna/__init__.py").is_file()
    assert (tmp_path / "urllib3/__init__.py").is_file()
    assert not (tmp_path / "tests").exists()
    assert not (tmp_path / ".venv").exists()
    assert not list(tmp_path.rglob("__pycache__"))
    assert not list(tmp_path.rglob("*.pyc"))
    assert not list(tmp_path.rglob("*.pyd"))


def test_document_ingestion_handler_imports_from_bundled_asset(
    tmp_path: Path,
):
    asset_path = tmp_path / "asset"
    stub_path = tmp_path / "runtime-stubs"
    asset_path.mkdir()
    stub_path.mkdir()
    (stub_path / "boto3.py").write_text(
        '"""Lambda-runtime boto3 placeholder for isolated import test."""\n',
        encoding="utf-8",
    )
    DocumentIngestionAssetBundler().try_bundle(str(asset_path), None)

    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import importlib, sys; "
                f"sys.path[:0] = [{str(asset_path)!r}, {str(stub_path)!r}]; "
                "importlib.import_module('lambda.document_ingestion.index')"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_document_ingestion_notification_is_raw_prefix_only():
    template = _template()
    notifications = template.find_resources(
        "Custom::S3BucketNotifications"
    )
    assert len(notifications) == 1
    configuration = next(iter(notifications.values()))["Properties"][
        "NotificationConfiguration"
    ]

    lambda_configurations = configuration[
        "LambdaFunctionConfigurations"
    ]
    assert len(lambda_configurations) == 1
    notification = lambda_configurations[0]
    assert notification["Events"] == ["s3:ObjectCreated:*"]
    assert notification["Filter"] == {
        "Key": {
            "FilterRules": [
                {
                    "Name": "prefix",
                    "Value": "knowledge/raw/",
                }
            ]
        }
    }
    assert all(
        output_prefix not in str(configuration)
        for output_prefix in (
            "knowledge/processed/",
            "knowledge/chunks/",
            "knowledge/embeddings/",
            "knowledge/metadata/",
            "knowledge/media/",
            "knowledge/quarantine/",
        )
    )


def test_knowledge_bucket_can_invoke_document_ingestion_lambda():
    template = _template()
    permissions = template.find_resources("AWS::Lambda::Permission")

    assert any(
        permission["Properties"].get("Principal") == "s3.amazonaws.com"
        and permission["Properties"].get("Action")
        == "lambda:InvokeFunction"
        and permission["Properties"].get("FunctionName")
        == {
            "Fn::GetAtt": [
                "DocumentIngestionFunction",
                "Arn",
            ]
        }
        for permission in permissions.values()
    )


def test_document_ingestion_role_has_prefix_scoped_permissions_only():
    template = _template()
    role, statements = _role_and_policy(
        template,
        "DocumentIngestionRole",
    )

    assert role["Properties"]["AssumeRolePolicyDocument"]["Statement"] == [
        {
            "Action": "sts:AssumeRole",
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
        }
    ]
    actions = _actions(statements)
    assert {
        "s3:ListBucket",
        "s3:GetObject",
        "s3:GetObjectVersion",
        "s3:PutObject",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "sqs:SendMessage",
    }.issubset(actions)
    assert "s3:*" not in actions
    assert "s3:ListAllMyBuckets" not in actions
    assert "s3:DeleteObject" not in actions
    assert "bedrock:InvokeModel" not in actions
    assert "secretsmanager:GetSecretValue" not in actions
    assert not any(action.startswith("ssm:") for action in actions)
    assert all(statement["Resource"] != "*" for statement in statements)

    list_statement = next(
        statement
        for statement in statements
        if statement["Action"] == "s3:ListBucket"
    )
    assert list_statement["Resource"] == {
        "Fn::GetAtt": ["KnowledgeBucketD7C8B00D", "Arn"]
    }
    assert set(
        list_statement["Condition"]["StringLike"]["s3:prefix"]
    ) == {
        "knowledge/embeddings/*",
        "knowledge/metadata/*",
    }
    assert len(
        list_statement["Condition"]["StringLike"]["s3:prefix"]
    ) == 2

    read_statement = next(
        statement
        for statement in statements
        if statement["Action"] == "s3:GetObject"
    )
    read_resources = read_statement["Resource"]
    assert isinstance(read_resources, list)
    assert {
        resource["Fn::Join"][1][1]
        for resource in read_resources
    } == {
        "/knowledge/embeddings/*",
        "/knowledge/metadata/*",
    }

    put_statement = next(
        statement
        for statement in statements
        if statement["Action"] == "s3:PutObject"
    )
    put_resources = str(put_statement["Resource"])
    assert {
        resource["Fn::Join"][1][1]
        for resource in put_statement["Resource"]
    } == {
        "/knowledge/processed/*",
        "/knowledge/chunks/*",
        "/knowledge/embeddings/*",
        "/knowledge/metadata/*",
        "/knowledge/media/*",
        "/knowledge/quarantine/*",
    }
    assert "knowledge/processed/*" in put_resources
    assert "knowledge/chunks/*" in put_resources
    assert "knowledge/embeddings/*" in put_resources
    assert "knowledge/metadata/*" in put_resources
    assert "knowledge/media/*" in put_resources
    assert "knowledge/quarantine/*" in put_resources
    assert "knowledge/raw/*" not in put_resources
    assert "knowledge/media/*" not in str(read_resources)
    assert "knowledge/quarantine/*" not in str(read_resources)


def test_document_ingestion_log_group_and_dead_letter_queue_are_bounded():
    template = _template()
    log_groups = template.find_resources("AWS::Logs::LogGroup")
    log_group = next(
        resource
        for resource in log_groups.values()
        if resource["Properties"].get("LogGroupName")
        == "/aws/lambda/dea-document-ingestion"
    )
    assert log_group["Properties"]["RetentionInDays"] == 30
    assert log_group["DeletionPolicy"] == "Delete"

    queues = template.find_resources("AWS::SQS::Queue")
    assert len(queues) == 1
    queue = next(iter(queues.values()))
    assert queue["Properties"]["QueueName"] == (
        "dea-document-ingestion-dlq"
    )
    assert queue["Properties"]["MessageRetentionPeriod"] == 14 * 24 * 60 * 60
    assert queue["Properties"]["SqsManagedSseEnabled"] is True
    assert queue["DeletionPolicy"] == "Delete"


def test_document_ingestion_resources_are_scoped_for_nonlegacy_clients():
    _, template = _stack_and_template(client_id="demo-client")
    function = template.find_resources("AWS::Lambda::Function")[
        "DocumentIngestionFunction"
    ]["Properties"]
    queue = next(
        iter(template.find_resources("AWS::SQS::Queue").values())
    )["Properties"]

    assert function["FunctionName"] == (
        "bah-de-assistant-demo-client-dev-document-ingestion"
    )
    assert function["Environment"]["Variables"]["CLIENT_ID"] == (
        "demo-client"
    )
    assert queue["QueueName"] == (
        "bah-de-assistant-demo-client-dev-document-ingestion-dlq"
    )
