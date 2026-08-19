import pytest

from config.clients import (
    DEFAULT_CLIENT_CONFIG_NAME,
    build_stack_id,
    get_client_config,
    normalize_context_value,
    resolve_client_config,
)


def test_default_configuration_is_internal_dev():
    config = resolve_client_config()

    assert DEFAULT_CLIENT_CONFIG_NAME == "internal-dev"
    assert config.client_id == "internal"
    assert config.environment == "dev"
    assert config.project == "bah-de-assistant"
    assert config.aws_region == "us-west-2"
    assert config.create_vector_bucket is False


def test_demo_client_configuration_resolves_with_cdk_file_defaults():
    config = resolve_client_config(
        {
            "client": "demo-client-dev",
            "project": "bah-de-assistant",
            "clientId": "internal",
            "environment": "dev",
            "createVectorBucket": False,
        }
    )

    assert config.client_id == "demo-client"
    assert config.environment == "dev"


def test_direct_context_can_override_selected_environment():
    config = resolve_client_config(
        {
            "client": "demo-client-dev",
            "environment": "test",
        }
    )

    assert config.client_id == "demo-client"
    assert config.environment == "test"


def test_unknown_client_configuration_raises_clear_error():
    with pytest.raises(
        ValueError,
        match="Unknown client configuration 'missing-client'",
    ):
        get_client_config("missing-client")


def test_invalid_environment_raises_clear_error():
    with pytest.raises(
        ValueError,
        match="Unsupported environment 'qa'",
    ):
        resolve_client_config({"environment": "qa"})


def test_explicit_aws_region_is_validated():
    assert resolve_client_config({"awsRegion": "us-east-1"}).aws_region == (
        "us-east-1"
    )
    with pytest.raises(ValueError, match="valid AWS Region"):
        resolve_client_config({"awsRegion": "not-a-region"})


def test_empty_normalized_context_value_raises_clear_error():
    with pytest.raises(
        ValueError,
        match="clientId must contain at least one letter or number",
    ):
        normalize_context_value(" --- ", "clientId")


def test_internal_dev_uses_legacy_stack_id():
    config = get_client_config("internal-dev")

    assert build_stack_id(config.client_id, config.environment) == (
        "DataEngineeringAssistantCdkStack"
    )


def test_demo_client_dev_uses_client_specific_stack_id():
    config = get_client_config("demo-client-dev")

    assert build_stack_id(config.client_id, config.environment) == (
        "DataEngineeringAssistant-Demo-Client-Dev-Stack"
    )


def test_named_client_configurations_have_isolated_stack_ids():
    internal = get_client_config("internal-dev")
    demo = get_client_config("demo-client-dev")

    internal_stack_id = build_stack_id(
        internal.client_id,
        internal.environment,
    )
    demo_stack_id = build_stack_id(
        demo.client_id,
        demo.environment,
    )

    assert internal_stack_id != demo_stack_id
