"""Typed client configuration and context normalization helpers."""

from dataclasses import dataclass, field
import re
from typing import Mapping
from urllib.parse import urlparse


DEFAULT_CLIENT_CONFIG_NAME = "internal-dev"
DEFAULT_PROJECT = "bah-de-assistant"
DEFAULT_CLIENT_ID = "internal"
DEFAULT_ENVIRONMENT = "dev"
DEFAULT_AWS_REGION = "us-west-2"
DEFAULT_CREATE_VECTOR_BUCKET = False
VALID_ENVIRONMENTS = frozenset({"dev", "test", "stage", "prod"})


@dataclass(frozen=True)
class ProductionIndexingConfig:
    """Opt-in, non-secret infrastructure settings for indexing."""

    enabled: bool = False
    embedding_provider: str = "bedrock"
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_dimensions: int | None = None
    bedrock_model_arn: str | None = None
    vector_store_provider: str = "qdrant"
    qdrant_endpoint_source: str = "environment"
    qdrant_url: str | None = None
    qdrant_collection: str | None = None
    qdrant_secret_arn: str | None = None
    qdrant_kms_key_arn: str | None = None
    dependency_layer_arn: str | None = None
    vpc_id: str | None = None
    subnet_ids: tuple[str, ...] = ()
    availability_zones: tuple[str, ...] = ()
    qdrant_security_group_id: str | None = None
    tls_required: bool = True
    connect_timeout_seconds: float = 5.0
    request_timeout_seconds: float = 10.0
    retry_limit: int = 2
    manifest_conflict_retries: int = 3
    maximum_descriptor_batch_size: int = 10
    maximum_chunks_per_invocation: int = 500
    reserved_concurrent_executions: int | None = None
    knowledge_namespace: str = "data-engineering"
    knowledge_domain: str = "data-engineering"

    def __post_init__(self) -> None:
        if self.enabled:
            _validate_production_indexing(self)


@dataclass(frozen=True)
class ClientConfig:
    """Configuration for one client and deployment environment."""

    client_id: str
    environment: str
    project: str
    create_vector_bucket: bool
    aws_region: str = DEFAULT_AWS_REGION
    production_indexing: ProductionIndexingConfig = field(
        default_factory=ProductionIndexingConfig
    )
    integration_validation_enabled: bool = False


CLIENT_CONFIGS: dict[str, ClientConfig] = {
    "internal-dev": ClientConfig(
        client_id="internal",
        environment="dev",
        project=DEFAULT_PROJECT,
        create_vector_bucket=False,
    ),
    "demo-client-dev": ClientConfig(
        client_id="demo-client",
        environment="dev",
        project=DEFAULT_PROJECT,
        create_vector_bucket=False,
    ),
}


def normalize_context_value(value: object, field_name: str) -> str:
    """Normalize a context value for use in physical resource names."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")

    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    if not normalized:
        raise ValueError(
            f"{field_name} must contain at least one letter or number after normalization"
        )
    return normalized


def normalize_environment(value: object) -> str:
    """Normalize and validate a deployment environment."""

    environment = normalize_context_value(value, "environment")
    if environment not in VALID_ENVIRONMENTS:
        supported = ", ".join(sorted(VALID_ENVIRONMENTS))
        raise ValueError(
            f"Unsupported environment '{environment}'. "
            f"Supported environments: {supported}"
        )
    return environment


def get_client_config(name: str) -> ClientConfig:
    """Return a named client configuration or raise a clear error."""

    normalized_name = name.strip().lower() if isinstance(name, str) else ""
    try:
        return CLIENT_CONFIGS[normalized_name]
    except KeyError as error:
        available = ", ".join(sorted(CLIENT_CONFIGS))
        raise ValueError(
            f"Unknown client configuration '{name}'. "
            f"Available configurations: {available}"
        ) from error


def _context_override(
    context: Mapping[str, object],
    key: str,
    selected_name: str,
    default_value: object,
) -> object | None:
    """Return a meaningful direct override without masking a named config.

    Values in cdk.json are visible through the same API as command-line
    context. Ignore those file defaults for non-default named configurations;
    a different value supplied with ``-c`` remains an override.
    """

    value = context.get(key)
    if value is None:
        return None
    if selected_name != DEFAULT_CLIENT_CONFIG_NAME and value == default_value:
        return None
    return value


def _normalize_boolean(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"{field_name} must be true or false")


def resolve_client_config(
    context: Mapping[str, object] | None = None,
) -> ClientConfig:
    """Resolve one named configuration plus supported direct overrides."""

    context = context or {}
    selected_value = context.get("client")
    selected_name = (
        DEFAULT_CLIENT_CONFIG_NAME
        if selected_value is None
        else str(selected_value).strip().lower()
    )
    selected = get_client_config(selected_name)

    project_override = _context_override(
        context,
        "project",
        selected_name,
        DEFAULT_PROJECT,
    )
    client_override = _context_override(
        context,
        "clientId",
        selected_name,
        DEFAULT_CLIENT_ID,
    )
    environment_override = _context_override(
        context,
        "environment",
        selected_name,
        DEFAULT_ENVIRONMENT,
    )
    vector_override = _context_override(
        context,
        "createVectorBucket",
        selected_name,
        DEFAULT_CREATE_VECTOR_BUCKET,
    )

    for forbidden in ("indexingQdrantApiKey", "qdrantApiKey"):
        if context.get(forbidden) not in {None, ""}:
            raise ValueError(
                f"{forbidden} is forbidden; use a Secrets Manager ARN"
            )
    indexing_enabled = _normalize_boolean(
        context.get("automaticIndexingEnabled", False),
        "automaticIndexingEnabled",
    )
    indexing = ProductionIndexingConfig(
        enabled=indexing_enabled,
        embedding_provider=str(
            context.get("indexingEmbeddingProvider", "bedrock")
        ).strip().lower(),
        embedding_model_id=str(
            context.get(
                "indexingEmbeddingModelId",
                "amazon.titan-embed-text-v2:0",
            )
        ).strip(),
        embedding_dimensions=_optional_context_integer(
            context, "indexingEmbeddingDimensions"
        ),
        bedrock_model_arn=_optional_context_text(
            context, "indexingBedrockModelArn"
        ),
        vector_store_provider=str(
            context.get("indexingVectorStoreProvider", "qdrant")
        ).strip().lower(),
        qdrant_endpoint_source=str(
            context.get("indexingQdrantEndpointSource", "environment")
        ).strip().lower(),
        qdrant_url=_optional_context_text(context, "indexingQdrantUrl"),
        qdrant_collection=_optional_context_text(
            context, "indexingQdrantCollection"
        ),
        qdrant_secret_arn=_optional_context_text(
            context, "indexingQdrantSecretArn"
        ),
        qdrant_kms_key_arn=_optional_context_text(
            context, "indexingQdrantKmsKeyArn"
        ),
        dependency_layer_arn=_optional_context_text(
            context, "indexingDependencyLayerArn"
        ),
        vpc_id=_optional_context_text(context, "indexingVpcId"),
        subnet_ids=_context_csv(context, "indexingSubnetIds"),
        availability_zones=_context_csv(
            context, "indexingAvailabilityZones"
        ),
        qdrant_security_group_id=_optional_context_text(
            context, "indexingQdrantSecurityGroupId"
        ),
        tls_required=_normalize_boolean(
            context.get("indexingTlsRequired", True),
            "indexingTlsRequired",
        ),
        connect_timeout_seconds=_context_float(
            context, "indexingConnectTimeoutSeconds", 5.0
        ),
        request_timeout_seconds=_context_float(
            context, "indexingRequestTimeoutSeconds", 10.0
        ),
        retry_limit=_context_integer(context, "indexingRetryLimit", 2),
        manifest_conflict_retries=_context_integer(
            context, "indexingManifestConflictRetries", 3
        ),
        maximum_descriptor_batch_size=_context_integer(
            context, "indexingMaximumDescriptorBatchSize", 10
        ),
        maximum_chunks_per_invocation=_context_integer(
            context, "indexingMaximumChunksPerInvocation", 500
        ),
        reserved_concurrent_executions=_optional_context_integer(
            context, "indexingReservedConcurrentExecutions"
        ),
        knowledge_namespace=str(
            context.get("indexingKnowledgeNamespace", "data-engineering")
        ).strip(),
        knowledge_domain=str(
            context.get("indexingKnowledgeDomain", "data-engineering")
        ).strip(),
    )
    resolved = ClientConfig(
        project=normalize_context_value(
            selected.project if project_override is None else project_override,
            "project",
        ),
        client_id=normalize_context_value(
            selected.client_id if client_override is None else client_override,
            "clientId",
        ),
        environment=normalize_environment(
            selected.environment
            if environment_override is None
            else environment_override
        ),
        create_vector_bucket=_normalize_boolean(
            selected.create_vector_bucket
            if vector_override is None
            else vector_override,
            "createVectorBucket",
        ),
        aws_region=normalize_aws_region(
            context.get("awsRegion", selected.aws_region)
        ),
        production_indexing=indexing,
        integration_validation_enabled=_normalize_boolean(
            context.get("integrationValidationEnabled", False),
            "integrationValidationEnabled",
        ),
    )
    if resolved.integration_validation_enabled:
        if selected_name != "internal-dev":
            raise ValueError(
                "Integration validation is restricted to internal-dev"
            )
        if resolved.environment == "prod":
            raise ValueError(
                "Integration validation cannot target production"
            )
        if not resolved.production_indexing.enabled:
            raise ValueError(
                "Integration validation requires automatic indexing enabled"
            )
    return resolved


def _optional_context_text(
    context: Mapping[str, object], key: str
) -> str | None:
    value = context.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _context_csv(
    context: Mapping[str, object], key: str
) -> tuple[str, ...]:
    raw_value = context.get(key)
    if isinstance(raw_value, (list, tuple)):
        if not all(isinstance(part, str) for part in raw_value):
            raise ValueError(f"{key} must contain only strings")
        return tuple(part.strip() for part in raw_value if part.strip())
    value = _optional_context_text(context, key)
    return (
        tuple(part.strip() for part in value.split(",") if part.strip())
        if value
        else ()
    )


def normalize_aws_region(value: object) -> str:
    """Validate an explicit AWS Region without changing its identity."""

    if not isinstance(value, str):
        raise ValueError("awsRegion must be a string")
    region = value.strip().lower()
    if not re.fullmatch(r"[a-z]{2}(?:-gov)?-[a-z]+-\d", region):
        raise ValueError("awsRegion must be a valid AWS Region identifier")
    return region


def _context_integer(
    context: Mapping[str, object], key: str, default: int
) -> int:
    try:
        value = int(context.get(key, default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be an integer") from error
    return value


def _optional_context_integer(
    context: Mapping[str, object], key: str
) -> int | None:
    value = context.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _context_integer(context, key, 0)


def _context_float(
    context: Mapping[str, object], key: str, default: float
) -> float:
    try:
        value = float(context.get(key, default))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be numeric") from error
    return value


def _validate_production_indexing(config: ProductionIndexingConfig) -> None:
    if config.embedding_provider != "bedrock":
        raise ValueError("Production indexing requires Bedrock")
    if config.vector_store_provider != "qdrant":
        raise ValueError("Production indexing requires Qdrant")
    if config.qdrant_endpoint_source not in {"environment", "secret"}:
        raise ValueError("Unsupported indexingQdrantEndpointSource")
    if not config.tls_required:
        raise ValueError("Production indexing requires TLS")
    if (
        config.connect_timeout_seconds <= 0
        or config.request_timeout_seconds <= 0
        or config.retry_limit < 0
        or config.manifest_conflict_retries <= 0
        or config.maximum_descriptor_batch_size <= 0
        or config.maximum_chunks_per_invocation <= 0
        or (
            config.reserved_concurrent_executions is not None
            and config.reserved_concurrent_executions <= 0
        )
        or config.embedding_dimensions is None
        or config.embedding_dimensions <= 0
    ):
        raise ValueError("Production indexing limits are invalid")
    if not config.knowledge_namespace or not config.knowledge_domain:
        raise ValueError("Production indexing namespace and domain are required")
    required = {
        "indexingBedrockModelArn": config.bedrock_model_arn,
        "indexingQdrantCollection": config.qdrant_collection,
        "indexingQdrantSecretArn": config.qdrant_secret_arn,
        "indexingDependencyLayerArn": config.dependency_layer_arn,
    }
    if config.qdrant_endpoint_source == "environment":
        required["indexingQdrantUrl"] = config.qdrant_url
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "Missing production indexing context: " + ", ".join(missing)
        )
    if config.qdrant_url:
        endpoint = urlparse(config.qdrant_url)
        if (
            endpoint.scheme != "https"
            or endpoint.hostname is None
            or endpoint.username
            or endpoint.password
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError("Production Qdrant endpoint must be safe HTTPS")
    if bool(config.vpc_id) != bool(config.subnet_ids):
        raise ValueError("indexingVpcId and indexingSubnetIds must be set together")
    if config.vpc_id and not config.qdrant_security_group_id:
        raise ValueError(
            "Private indexing requires indexingQdrantSecurityGroupId"
        )
    if config.subnet_ids and len(config.subnet_ids) != len(config.availability_zones):
        raise ValueError("Each indexing subnet requires an availability zone")


def build_stack_id(client_id: str, environment: str) -> str:
    """Build the deterministic client/environment-specific stack identity."""

    normalized_client = normalize_context_value(client_id, "clientId")
    normalized_environment = normalize_environment(environment)
    if (
        normalized_client == DEFAULT_CLIENT_ID
        and normalized_environment == DEFAULT_ENVIRONMENT
    ):
        return "DataEngineeringAssistantCdkStack"

    return (
        f"DataEngineeringAssistant-{normalized_client.title()}-"
        f"{normalized_environment.title()}-Stack"
    )
