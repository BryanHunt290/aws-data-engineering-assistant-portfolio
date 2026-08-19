import json
import logging
import os

import boto3
from botocore.exceptions import ClientError


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _response(status_code: int, body: dict) -> dict:
    """Build an API-compatible Lambda response without requiring API Gateway."""
    return {
        "statusCode": status_code,
        "body": json.dumps(body),
    }


def handler(event, context):
    """Check that the configured Glue database name is readable from SSM."""
    del event, context

    try:
        parameter_name = os.environ["GLUE_DATABASE_PARAMETER"]
        ssm_client = boto3.client("ssm")
        response = ssm_client.get_parameter(
            Name=parameter_name,
            WithDecryption=False,
        )
        database_name = response["Parameter"]["Value"]

        if not database_name:
            raise ValueError("The Glue database parameter is empty")

        # The catalog database name is operational metadata, not a secret. Do
        # not log the complete SSM response or any other parameter values.
        logger.info("Retrieved Glue database name: %s", database_name)

        return _response(
            200,
            {
                "service": "data-engineering-assistant",
                "status": "healthy",
                "glue_database": database_name,
            },
        )
    except ClientError:
        # Avoid logging the AWS exception payload because it can contain request
        # details. Callers receive a generic health response instead.
        logger.error("Unable to retrieve the Glue database parameter from SSM")
    except (KeyError, TypeError, ValueError):
        logger.error("The Glue database health-check configuration is invalid")
    except Exception:
        # Keep unexpected exception details out of logs to avoid accidental
        # disclosure while still returning a stable failure response.
        logger.error("The Glue database health check failed unexpectedly")

    return _response(
        500,
        {
            "service": "data-engineering-assistant",
            "status": "unhealthy",
        },
    )
