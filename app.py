#!/usr/bin/env python3
import aws_cdk as cdk

from config.clients import build_stack_id, resolve_client_config
from data_engineering_assistant_cdk.data_engineering_assistant_cdk_stack import DataEngineeringAssistantCdkStack


app = cdk.App()
client_config = resolve_client_config(app.node.get_all_context())
stack_id = build_stack_id(
    client_config.client_id,
    client_config.environment,
)

DataEngineeringAssistantCdkStack(
    app,
    stack_id,
    project=client_config.project,
    client_id=client_config.client_id,
    environment=client_config.environment,
    create_vector_bucket=client_config.create_vector_bucket,
    production_indexing=client_config.production_indexing,
    # Specialize this stack for the requested deployment region while leaving
    # the AWS account unresolved so the active CDK credentials choose it.
    env=cdk.Environment(region=client_config.aws_region),
)

app.synth()
