import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import Mock, patch


# Lambda provides boto3 and botocore at runtime. Stub those modules before
# loading the handler so local unit tests do not need to package the AWS SDK.
boto3_stub = ModuleType("boto3")
boto3_stub.client = Mock()
botocore_stub = ModuleType("botocore")
botocore_exceptions_stub = ModuleType("botocore.exceptions")


class ClientErrorStub(Exception):
    pass


botocore_exceptions_stub.ClientError = ClientErrorStub
botocore_stub.exceptions = botocore_exceptions_stub

HANDLER_PATH = (
    Path(__file__).parents[2] / "lambda" / "health_check" / "index.py"
)
HANDLER_SPEC = importlib.util.spec_from_file_location(
    "health_check_handler",
    HANDLER_PATH,
)
health_check = importlib.util.module_from_spec(HANDLER_SPEC)
with patch.dict(
    sys.modules,
    {
        "boto3": boto3_stub,
        "botocore": botocore_stub,
        "botocore.exceptions": botocore_exceptions_stub,
    },
):
    HANDLER_SPEC.loader.exec_module(health_check)


def test_handler_returns_healthy_response_with_database_name():
    ssm_client = Mock()
    ssm_client.get_parameter.return_value = {
        "Parameter": {"Value": "dea_catalog"}
    }

    with (
        patch.object(health_check.boto3, "client", return_value=ssm_client),
        patch.object(health_check.logger, "info") as log_info,
        patch.dict(
            health_check.os.environ,
            {"GLUE_DATABASE_PARAMETER": "/dea/glue/database"},
        ),
    ):
        response = health_check.handler({}, None)

    assert response["statusCode"] == 200
    assert json.loads(response["body"]) == {
        "service": "data-engineering-assistant",
        "status": "healthy",
        "glue_database": "dea_catalog",
    }
    ssm_client.get_parameter.assert_called_once_with(
        Name="/dea/glue/database",
        WithDecryption=False,
    )
    log_info.assert_called_once_with(
        "Retrieved Glue database name: %s",
        "dea_catalog",
    )


def test_handler_returns_generic_failure_without_exception_details():
    ssm_client = Mock()
    ssm_client.get_parameter.return_value = {"Parameter": {}}

    with (
        patch.object(health_check.boto3, "client", return_value=ssm_client),
        patch.object(health_check.logger, "error") as log_error,
        patch.dict(
            health_check.os.environ,
            {"GLUE_DATABASE_PARAMETER": "/dea/glue/database"},
        ),
    ):
        response = health_check.handler({}, None)

    assert response["statusCode"] == 500
    assert json.loads(response["body"]) == {
        "service": "data-engineering-assistant",
        "status": "unhealthy",
    }
    log_error.assert_called_once_with(
        "The Glue database health-check configuration is invalid"
    )
