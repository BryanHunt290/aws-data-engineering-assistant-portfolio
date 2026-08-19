from aws_cdk import (
    ArnFormat,
    AssetHashType,
    Aws,
    BundlingOptions,
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    Tags,
    aws_athena as athena,
    aws_ec2 as ec2,
    aws_glue as glue,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_logs as logs,
    aws_s3 as s3,
    aws_s3_notifications as s3_notifications,
    aws_sqs as sqs,
    aws_ssm as ssm,
)
from constructs import Construct

from config.clients import (
    DEFAULT_CLIENT_ID,
    DEFAULT_CREATE_VECTOR_BUCKET,
    DEFAULT_ENVIRONMENT,
    DEFAULT_PROJECT,
    ProductionIndexingConfig,
    normalize_context_value,
    normalize_environment,
)
from data_engineering_assistant_cdk.lambda_asset import (
    DocumentIngestionAssetBundler,
)


class DataEngineeringAssistantCdkStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        project: str = DEFAULT_PROJECT,
        client_id: str = DEFAULT_CLIENT_ID,
        environment: str = DEFAULT_ENVIRONMENT,
        create_vector_bucket: bool = DEFAULT_CREATE_VECTOR_BUCKET,
        production_indexing: ProductionIndexingConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        project = normalize_context_value(project, "project")
        client_id = normalize_context_value(client_id, "clientId")
        environment = normalize_environment(environment)
        resource_prefix = f"{project}-{client_id}-{environment}"
        indexing = production_indexing or ProductionIndexingConfig()
        collection_scope = f"{client_id}_{environment}".replace("-", "_")
        if indexing.enabled and collection_scope not in (
            indexing.qdrant_collection or ""
        ):
            raise ValueError(
                "Production Qdrant collection must include client and environment"
            )

        self.project = project
        self.client_id = client_id
        self.deployment_environment = environment
        self.resource_prefix = resource_prefix
        # The current local stack has no S3 Vector construct. Retain the context
        # value so that the existing project switch remains available when that
        # optional resource is introduced or restored.
        self.create_vector_bucket = create_vector_bucket

        Tags.of(self).add("Project", "data-engineering-assistant")
        Tags.of(self).add("ClientId", client_id)
        Tags.of(self).add("Environment", environment)
        Tags.of(self).add("ManagedBy", "aws-cdk")
        Tags.of(self).add("Owner", "bryan")

        is_legacy_internal_deployment = (
            project == DEFAULT_PROJECT
            and client_id == DEFAULT_CLIENT_ID
            and environment == DEFAULT_ENVIRONMENT
        )

        def bucket_name(layer: str) -> str | None:
            # Existing internal-dev buckets use CloudFormation-generated names.
            # Adding BucketName now would replace deployed buckets, so preserve
            # that legacy formula while giving future client stacks unique names.
            if is_legacy_internal_deployment:
                return None
            return (
                f"{resource_prefix}-{layer}-{Aws.REGION}-{Aws.ACCOUNT_ID}"
            )

        parameter_root = (
            "/dea"
            if is_legacy_internal_deployment
            else f"/{resource_prefix}"
        )
        glue_database_name = (
            "dea_catalog"
            if is_legacy_internal_deployment
            else f"{resource_prefix}-catalog"
        )
        athena_workgroup_name = (
            "dea-workgroup"
            if is_legacy_internal_deployment
            else f"{resource_prefix}-workgroup"
        )
        health_check_function_name = (
            "dea-health-check"
            if is_legacy_internal_deployment
            else f"{resource_prefix}-health-check"
        )
        document_ingestion_function_name = (
            "dea-document-ingestion"
            if is_legacy_internal_deployment
            else f"{resource_prefix}-document-ingestion"
        )
        document_ingestion_dlq_name = (
            "dea-document-ingestion-dlq"
            if is_legacy_internal_deployment
            else f"{resource_prefix}-document-ingestion-dlq"
        )

        # Apply the security settings required for every data bucket. Omitting
        # bucket_name intentionally lets CloudFormation generate a unique name.
        secure_bucket_defaults = {
            "encryption": s3.BucketEncryption.S3_MANAGED,
            "block_public_access": s3.BlockPublicAccess.BLOCK_ALL,
            "enforce_ssl": True,
        }

        # Retain source data and its history if the stack is deleted.
        raw_bucket = s3.Bucket(
            self,
            "RawBucket",
            bucket_name=bucket_name("raw"),
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
            # Move raw objects to Standard-IA after 90 days to reduce storage cost.
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="TransitionToInfrequentAccess",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(90),
                        )
                    ],
                )
            ],
            **secure_bucket_defaults,
        )

        # Retain versioned, transformed data if the stack is deleted.
        curated_bucket = s3.Bucket(
            self,
            "CuratedBucket",
            bucket_name=bucket_name("curated"),
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
            **secure_bucket_defaults,
        )

        # Retain versioned knowledge-base content if the stack is deleted.
        knowledge_bucket = s3.Bucket(
            self,
            "KnowledgeBucket",
            bucket_name=bucket_name("kb"),
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
            **secure_bucket_defaults,
        )

        # Retain versioned model artifacts if the stack is deleted.
        models_bucket = s3.Bucket(
            self,
            "ModelsBucket",
            bucket_name=bucket_name("models"),
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
            **secure_bucket_defaults,
        )

        # Logs are temporary development data, so keep versioning disabled,
        # expire objects after 90 days, and empty/delete the bucket with the stack.
        logs_bucket = s3.Bucket(
            self,
            "LogsBucket",
            bucket_name=bucket_name("logs"),
            versioned=False,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="DeleteAfter90Days",
                    expiration=Duration.days(90),
                )
            ],
            **secure_bucket_defaults,
        )

        # Athena query results are also disposable development data; expire them
        # after 30 days and empty/delete the unversioned bucket with the stack.
        athena_results_bucket = s3.Bucket(
            self,
            "AthenaResultsBucket",
            bucket_name=bucket_name("athena"),
            versioned=False,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="DeleteAfter30Days",
                    expiration=Duration.days(30),
                )
            ],
            **secure_bucket_defaults,
        )

        # Create the shared Glue Data Catalog database used by the assistant.
        data_engineering_database = glue.CfnDatabase(
            self,
            "DataEngineeringDatabase",
            catalog_id=Aws.ACCOUNT_ID,
            database_input=glue.CfnDatabase.DatabaseInputProperty(
                name=glue_database_name,
                description="Data catalog for the Data Engineering Assistant",
            ),
        )

        # Configure Athena to place every query result in the generated results
        # bucket and prevent clients from overriding the workgroup settings.
        athena_workgroup = athena.CfnWorkGroup(
            self,
            "DataEngineeringWorkgroup",
            name=athena_workgroup_name,
            description="Athena workgroup for the Data Engineering Assistant",
            state="ENABLED",
            work_group_configuration=athena.CfnWorkGroup.WorkGroupConfigurationProperty(
                enforce_work_group_configuration=True,
                result_configuration=athena.CfnWorkGroup.ResultConfigurationProperty(
                    output_location=athena_results_bucket.s3_url_for_object(
                        "query-results/"
                    ),
                    encryption_configuration=athena.CfnWorkGroup.EncryptionConfigurationProperty(
                        encryption_option="SSE_S3"
                    ),
                ),
            ),
        )

        # Publish resource identifiers through Parameter Store so applications
        # can discover them without duplicating generated bucket names.
        parameter_values = {
            "GlueDatabaseParameter": (
                f"{parameter_root}/glue/database",
                data_engineering_database.ref,
            ),
            "AthenaWorkgroupParameter": (
                f"{parameter_root}/athena/workgroup",
                athena_workgroup.ref,
            ),
            "RawBucketParameter": (
                f"{parameter_root}/buckets/raw",
                raw_bucket.bucket_name,
            ),
            "CuratedBucketParameter": (
                f"{parameter_root}/buckets/curated",
                curated_bucket.bucket_name,
            ),
            "KnowledgeBucketParameter": (
                f"{parameter_root}/buckets/knowledge",
                knowledge_bucket.bucket_name,
            ),
            "ModelsBucketParameter": (
                f"{parameter_root}/buckets/models",
                models_bucket.bucket_name,
            ),
            "LogsBucketParameter": (
                f"{parameter_root}/buckets/logs",
                logs_bucket.bucket_name,
            ),
            "AthenaResultsBucketParameter": (
                f"{parameter_root}/buckets/athena-results",
                athena_results_bucket.bucket_name,
            ),
        }

        resource_parameters = []
        for parameter_id, (parameter_name, parameter_value) in parameter_values.items():
            resource_parameters.append(
                ssm.StringParameter(
                    self,
                    parameter_id,
                    parameter_name=parameter_name,
                    string_value=parameter_value,
                    description=(
                        f"Data Engineering Assistant value for {parameter_name}"
                    ),
                )
            )

        # Allow Glue jobs to assume this role without assigning a fixed physical
        # role name, so CloudFormation can generate an account-unique name.
        glue_execution_role = iam.Role(
            self,
            "GlueExecutionRole",
            assumed_by=iam.ServicePrincipal("glue.amazonaws.com"),
            description="Least-privilege execution role for Glue ETL jobs",
        )

        # Glue can list both data buckets and resolve their regions.
        glue_execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:GetBucketLocation",
                    "s3:ListBucket",
                ],
                resources=[
                    raw_bucket.bucket_arn,
                    curated_bucket.bucket_arn,
                ],
            )
        )

        # Raw is read-only input, including access to its versioned objects.
        glue_execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:GetObject",
                    "s3:GetObjectVersion",
                ],
                resources=[raw_bucket.arn_for_objects("*")],
            )
        )

        # Curated is the ETL output. Permit reading existing versions and writing
        # transformed objects, without granting bucket administration.
        glue_execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:GetObject",
                    "s3:GetObjectVersion",
                    "s3:PutObject",
                ],
                resources=[curated_bucket.arn_for_objects("*")],
            )
        )

        # Limit Glue Data Catalog access to this account's catalog, the assistant
        # database, and tables or partitions contained in that database.
        glue_execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "glue:GetDatabase",
                    "glue:GetDatabases",
                    "glue:GetTable",
                    "glue:GetTables",
                    "glue:GetTableVersion",
                    "glue:GetTableVersions",
                    "glue:CreateTable",
                    "glue:UpdateTable",
                    "glue:GetPartition",
                    "glue:GetPartitions",
                    "glue:BatchGetPartition",
                    "glue:CreatePartition",
                    "glue:BatchCreatePartition",
                    "glue:UpdatePartition",
                ],
                resources=[
                    Stack.of(self).format_arn(
                        service="glue",
                        resource="catalog",
                    ),
                    Stack.of(self).format_arn(
                        service="glue",
                        resource="database",
                        resource_name=data_engineering_database.ref,
                        arn_format=ArnFormat.SLASH_RESOURCE_NAME,
                    ),
                    Stack.of(self).format_arn(
                        service="glue",
                        resource="table",
                        resource_name=f"{data_engineering_database.ref}/*",
                        arn_format=ArnFormat.SLASH_RESOURCE_NAME,
                    ),
                ],
            )
        )

        # Permit only creation and writes for the standard Glue log-group prefix.
        glue_execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=["logs:CreateLogGroup"],
                resources=[
                    Stack.of(self).format_arn(
                        service="logs",
                        resource="log-group",
                        resource_name="/aws-glue/*",
                        arn_format=ArnFormat.COLON_RESOURCE_NAME,
                    )
                ],
            )
        )
        glue_execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                resources=[
                    Stack.of(self).format_arn(
                        service="logs",
                        resource="log-group",
                        resource_name="/aws-glue/*:log-stream:*",
                        arn_format=ArnFormat.COLON_RESOURCE_NAME,
                    )
                ],
            )
        )

        # Allow Lambda functions to assume a separate role with access only to
        # the assistant resources needed for retrieval and model invocation.
        lambda_execution_role = iam.Role(
            self,
            "LambdaExecutionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Least-privilege execution role for assistant Lambdas",
        )

        # Lambda can list and retrieve versioned knowledge documents but cannot
        # modify the bucket or its objects.
        lambda_execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:GetBucketLocation",
                    "s3:ListBucket",
                ],
                resources=[knowledge_bucket.bucket_arn],
            )
        )
        lambda_execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:GetObject",
                    "s3:GetObjectVersion",
                ],
                resources=[knowledge_bucket.arn_for_objects("*")],
            )
        )

        # Grant only current-value reads for the eight parameters in this stack.
        lambda_execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "ssm:GetParameter",
                    "ssm:GetParameters",
                ],
                resources=[
                    resource_parameter.parameter_arn
                    for resource_parameter in resource_parameters
                ],
            )
        )

        # No model ID was specified, so allow only InvokeModel and scope it to
        # Bedrock foundation-model resources in this stack's region.
        lambda_execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=[
                    Stack.of(self).format_arn(
                        service="bedrock",
                        account="",
                        resource="foundation-model",
                        resource_name="*",
                        arn_format=ArnFormat.SLASH_RESOURCE_NAME,
                    )
                ],
            )
        )

        # Permit only creation and writes for the standard Lambda log-group prefix.
        lambda_execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=["logs:CreateLogGroup"],
                resources=[
                    Stack.of(self).format_arn(
                        service="logs",
                        resource="log-group",
                        resource_name="/aws/lambda/*",
                        arn_format=ArnFormat.COLON_RESOURCE_NAME,
                    )
                ],
            )
        )
        lambda_execution_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                resources=[
                    Stack.of(self).format_arn(
                        service="logs",
                        resource="log-group",
                        resource_name="/aws/lambda/*:log-stream:*",
                        arn_format=ArnFormat.COLON_RESOURCE_NAME,
                    )
                ],
            )
        )

        # Create the health-check log group explicitly so retention and
        # development cleanup behavior are controlled by this stack.
        health_check_log_group = logs.LogGroup(
            self,
            "HealthCheckLogGroup",
            log_group_name=f"/aws/lambda/{health_check_function_name}",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Run the first health check from the local Lambda asset using the
        # existing least-privilege Lambda execution role.
        health_check_function = lambda_.Function(
            self,
            "HealthCheckFunction",
            function_name=health_check_function_name,
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="index.handler",
            code=lambda_.Code.from_asset("lambda/health_check"),
            role=lambda_execution_role,
            timeout=Duration.seconds(30),
            memory_size=256,
            environment={
                "GLUE_DATABASE_PARAMETER": f"{parameter_root}/glue/database",
            },
            log_group=health_check_log_group,
        )

        # The L2 Function normally adds a hash to its synthesized resource ID;
        # override it to preserve the requested CloudFormation logical ID.
        health_check_function.node.default_child.override_logical_id(
            "HealthCheckFunction"
        )

        # This dedicated role receives provider permissions only when indexing
        # is explicitly enabled.
        document_ingestion_role = iam.Role(
            self,
            "DocumentIngestionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description=(
                "Least-privilege role for S3 knowledge document ingestion"
            ),
        )
        document_ingestion_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[knowledge_bucket.bucket_arn],
                conditions={
                    "StringLike": {
                        "s3:prefix": [
                            "knowledge/metadata/*",
                            "knowledge/embeddings/*",
                        ]
                    }
                },
            )
        )

        if indexing.enabled:
            document_ingestion_role.add_to_policy(
                iam.PolicyStatement(
                    actions=["bedrock:InvokeModel"],
                    resources=[indexing.bedrock_model_arn],
                )
            )
            document_ingestion_role.add_to_policy(
                iam.PolicyStatement(
                    actions=["secretsmanager:GetSecretValue"],
                    resources=[indexing.qdrant_secret_arn],
                )
            )
            if indexing.qdrant_kms_key_arn:
                document_ingestion_role.add_to_policy(
                    iam.PolicyStatement(
                        actions=["kms:Decrypt"],
                        resources=[indexing.qdrant_kms_key_arn],
                    )
                )
        document_ingestion_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:GetObject",
                    "s3:GetObjectVersion",
                ],
                resources=[
                    knowledge_bucket.arn_for_objects("knowledge/raw/*"),
                ],
            )
        )
        indexing_read_resources = [
            knowledge_bucket.arn_for_objects("knowledge/metadata/*"),
            knowledge_bucket.arn_for_objects("knowledge/embeddings/*"),
        ]
        if indexing.enabled:
            indexing_read_resources.append(
                knowledge_bucket.arn_for_objects("knowledge/chunks/*")
            )
        document_ingestion_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=indexing_read_resources,
            )
        )
        document_ingestion_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:PutObject"],
                resources=[
                    knowledge_bucket.arn_for_objects(
                        "knowledge/processed/*"
                    ),
                    knowledge_bucket.arn_for_objects("knowledge/chunks/*"),
                    knowledge_bucket.arn_for_objects(
                        "knowledge/embeddings/*"
                    ),
                    knowledge_bucket.arn_for_objects(
                        "knowledge/metadata/*"
                    ),
                    knowledge_bucket.arn_for_objects(
                        "knowledge/media/*"
                    ),
                    knowledge_bucket.arn_for_objects(
                        "knowledge/quarantine/*"
                    ),
                ],
            )
        )

        document_ingestion_log_group = logs.LogGroup(
            self,
            "DocumentIngestionLogGroup",
            log_group_name=(
                f"/aws/lambda/{document_ingestion_function_name}"
            ),
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )
        document_ingestion_log_group.grant_write(
            document_ingestion_role
        )

        # S3 invokes Lambda asynchronously. This queue is the Lambda failure
        # sink after asynchronous retries; it is not between S3 and Lambda.
        document_ingestion_dead_letter_queue = sqs.Queue(
            self,
            "DocumentIngestionDeadLetterQueue",
            queue_name=document_ingestion_dlq_name,
            encryption=sqs.QueueEncryption.SQS_MANAGED,
            enforce_ssl=True,
            retention_period=Duration.days(14),
            removal_policy=RemovalPolicy.DESTROY,
        )

        ingestion_environment = {
            "CLIENT_ID": client_id,
            "DEPLOYMENT_ENVIRONMENT": environment,
            "KNOWLEDGE_BUCKET_NAME": knowledge_bucket.bucket_name,
            "KNOWLEDGE_CHUNK_OVERLAP": "100",
            "KNOWLEDGE_CHUNK_SIZE": "1000",
            "KNOWLEDGE_AUTOMATIC_INDEXING_ENABLED": str(
                indexing.enabled
            ).lower(),
            "KNOWLEDGE_DOMAIN": "data-engineering",
            "KNOWLEDGE_MAXIMUM_UPLOAD_SIZE": str(10 * 1024 * 1024),
            "KNOWLEDGE_NAMESPACE": "data-engineering",
            "KNOWLEDGE_RAW_PREFIX": "knowledge/raw/",
            "KNOWLEDGE_SUPPORTED_DOCUMENT_TYPES": (
                "html,json,markdown,md,pdf,py,txt"
            ),
        }
        layers = None
        function_network: dict[str, object] = {}
        if indexing.enabled:
            ingestion_environment.update(
                {
                    "KNOWLEDGE_INDEXING_RUNTIME_MODE": "production",
                    "KNOWLEDGE_EMBEDDING_PROVIDER": indexing.embedding_provider,
                    "KNOWLEDGE_EMBEDDING_MODEL_ID": indexing.embedding_model_id,
                    "KNOWLEDGE_EMBEDDING_DIMENSIONS": str(
                        indexing.embedding_dimensions
                    ),
                    "KNOWLEDGE_VECTOR_STORE_PROVIDER": indexing.vector_store_provider,
                    "KNOWLEDGE_QDRANT_ENDPOINT_SOURCE": indexing.qdrant_endpoint_source,
                    "KNOWLEDGE_QDRANT_COLLECTION": indexing.qdrant_collection,
                    "KNOWLEDGE_QDRANT_SECRET_IDENTIFIER": indexing.qdrant_secret_arn,
                    "KNOWLEDGE_QDRANT_TLS_REQUIRED": "true",
                    "KNOWLEDGE_QDRANT_AUTHENTICATION_REQUIRED": "true",
                    "KNOWLEDGE_CONNECT_TIMEOUT_SECONDS": str(indexing.connect_timeout_seconds),
                    "KNOWLEDGE_REQUEST_TIMEOUT_SECONDS": str(indexing.request_timeout_seconds),
                    "KNOWLEDGE_INDEXING_RETRY_LIMIT": str(indexing.retry_limit),
                    "KNOWLEDGE_MANIFEST_CONFLICT_RETRIES": str(indexing.manifest_conflict_retries),
                    "KNOWLEDGE_MAX_DESCRIPTOR_BATCH_SIZE": str(indexing.maximum_descriptor_batch_size),
                    "KNOWLEDGE_MAX_CHUNKS_PER_INVOCATION": str(indexing.maximum_chunks_per_invocation),
                    "KNOWLEDGE_NAMESPACE": indexing.knowledge_namespace,
                    "KNOWLEDGE_DOMAIN": indexing.knowledge_domain,
                }
            )
            if indexing.qdrant_url:
                ingestion_environment["KNOWLEDGE_QDRANT_URL"] = indexing.qdrant_url
            layers = [
                lambda_.LayerVersion.from_layer_version_arn(
                    self,
                    "IndexingDependencyLayer",
                    indexing.dependency_layer_arn,
                )
            ]
            if indexing.vpc_id:
                imported_subnets = [
                    ec2.Subnet.from_subnet_attributes(
                        self,
                        f"IndexingSubnet{position}",
                        subnet_id=subnet_id,
                        availability_zone=indexing.availability_zones[position],
                    )
                    for position, subnet_id in enumerate(indexing.subnet_ids)
                ]
                vpc = ec2.Vpc.from_vpc_attributes(
                    self,
                    "IndexingVpc",
                    vpc_id=indexing.vpc_id,
                    availability_zones=list(indexing.availability_zones),
                    private_subnet_ids=list(indexing.subnet_ids),
                )
                lambda_sg = ec2.SecurityGroup(
                    self,
                    "IndexingLambdaSecurityGroup",
                    vpc=vpc,
                    allow_all_outbound=False,
                    description="Least-privilege egress for production indexing",
                )
                qdrant_sg = ec2.SecurityGroup.from_security_group_id(
                    self,
                    "QdrantSecurityGroup",
                    indexing.qdrant_security_group_id,
                )
                lambda_sg.add_egress_rule(qdrant_sg, ec2.Port.tcp(443))
                endpoint_sg = ec2.SecurityGroup(
                    self,
                    "IndexingEndpointSecurityGroup",
                    vpc=vpc,
                    allow_all_outbound=False,
                    description="TLS access to indexing AWS service endpoints",
                )
                endpoint_sg.add_ingress_rule(lambda_sg, ec2.Port.tcp(443))
                lambda_sg.add_egress_rule(endpoint_sg, ec2.Port.tcp(443))
                endpoint_selection = ec2.SubnetSelection(
                    subnets=imported_subnets
                )
                for endpoint_id, service in (
                    (
                        "IndexingSecretsEndpoint",
                        ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER,
                    ),
                    (
                        "IndexingLogsEndpoint",
                        ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS,
                    ),
                    (
                        "IndexingBedrockEndpoint",
                        ec2.InterfaceVpcEndpointAwsService.BEDROCK_RUNTIME,
                    ),
                    (
                        "IndexingS3Endpoint",
                        ec2.InterfaceVpcEndpointAwsService.S3,
                    ),
                ):
                    ec2.InterfaceVpcEndpoint(
                        self,
                        endpoint_id,
                        vpc=vpc,
                        service=service,
                        subnets=endpoint_selection,
                        security_groups=[endpoint_sg],
                        open=False,
                    )
                function_network = {
                    "vpc": vpc,
                    "vpc_subnets": ec2.SubnetSelection(subnets=imported_subnets),
                    "security_groups": [lambda_sg],
                }

        function_scaling = {}
        if indexing.reserved_concurrent_executions is not None:
            function_scaling["reserved_concurrent_executions"] = (
                indexing.reserved_concurrent_executions
            )

        document_ingestion_function = lambda_.Function(
            self,
            "DocumentIngestionFunction",
            function_name=document_ingestion_function_name,
            description=(
                "Processes ObjectCreated events from the knowledge raw prefix"
            ),
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="lambda.document_ingestion.index.handler",
            code=lambda_.Code.from_asset(
                ".",
                asset_hash_type=AssetHashType.OUTPUT,
                bundling=BundlingOptions(
                    image=lambda_.Runtime.PYTHON_3_12.bundling_image,
                    local=DocumentIngestionAssetBundler(),
                ),
            ),
            role=document_ingestion_role,
            timeout=Duration.minutes(5),
            memory_size=512,
            dead_letter_queue=document_ingestion_dead_letter_queue,
            environment=ingestion_environment,
            layers=layers,
            **function_scaling,
            **function_network,
            log_group=document_ingestion_log_group,
        )
        document_ingestion_function.node.default_child.override_logical_id(
            "DocumentIngestionFunction"
        )

        # A prefix-only notification avoids duplicate suffix configurations and
        # ensures generated output prefixes cannot invoke this function.
        knowledge_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3_notifications.LambdaDestination(
                document_ingestion_function
            ),
            s3.NotificationKeyFilter(prefix="knowledge/raw/"),
        )

        # Export generated physical names so applications and operators can find
        # every bucket without relying on hard-coded globally unique names.
        bucket_outputs = {
            "RawBucketName": raw_bucket.bucket_name,
            "CuratedBucketName": curated_bucket.bucket_name,
            "KnowledgeBucketName": knowledge_bucket.bucket_name,
            "ModelsBucketName": models_bucket.bucket_name,
            "LogsBucketName": logs_bucket.bucket_name,
            "AthenaResultsBucketName": athena_results_bucket.bucket_name,
        }

        for output_id, bucket_name in bucket_outputs.items():
            CfnOutput(
                self,
                output_id,
                value=bucket_name,
                description=f"Generated name of the {output_id.removesuffix('BucketName')} bucket",
            )

        # Export the catalog database and workgroup names alongside bucket names.
        CfnOutput(
            self,
            "GlueDatabaseName",
            value=data_engineering_database.ref,
            description="Name of the Data Engineering Assistant Glue database",
        )
        CfnOutput(
            self,
            "AthenaWorkgroupName",
            value=athena_workgroup.ref,
            description="Name of the Data Engineering Assistant Athena workgroup",
        )

        # Export generated IAM role ARNs for use by jobs and functions.
        CfnOutput(
            self,
            "GlueExecutionRoleArn",
            value=glue_execution_role.role_arn,
            description="ARN of the Glue execution role",
        )
        CfnOutput(
            self,
            "LambdaExecutionRoleArn",
            value=lambda_execution_role.role_arn,
            description="ARN of the Lambda execution role",
        )

        # Export the health-check function's fixed operational name.
        CfnOutput(
            self,
            "HealthCheckFunctionName",
            value=health_check_function.function_name,
            description="Name of the Data Engineering Assistant health check",
        )

        CfnOutput(
            self,
            "ClientId",
            value=client_id,
            description="Client identifier selected for this stack",
        )
        CfnOutput(
            self,
            "DeploymentEnvironment",
            value=environment,
            description="Deployment environment selected for this stack",
        )
        CfnOutput(
            self,
            "ResourcePrefix",
            value=resource_prefix,
            description="Reusable prefix for client-aware physical resource names",
        )
