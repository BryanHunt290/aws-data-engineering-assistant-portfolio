# AWS infrastructure

This document describes only the resources currently implemented by the AWS
CDK application. The stack includes a bounded S3-triggered document-ingestion
runtime. Retrieval, RAG, and Streamlit remain application code; the stack does
not add a managed vector store, hosted Streamlit service, or API.

## Stack selection and compatibility

One CDK context value selects one client configuration:

```powershell
cdk.cmd synth -c client=internal-dev
cdk.cmd synth -c client=demo-client-dev
```

The deployed internal development environment retains the exact legacy CDK
stack ID and CloudFormation stack identity:

```text
DataEngineeringAssistantCdkStack
```

This compatibility rule prevents CDK from treating the existing environment as
a different stack. Non-legacy clients receive scoped IDs; the included example
uses:

```text
DataEngineeringAssistant-Demo-Client-Dev-Stack
```

The app is specialized to `us-west-2`. Its AWS account is unresolved at synth
time, so the active CDK credentials determine the account for account-aware
operations. Configuration files contain no account ID or credentials.

## Resource inventory

The stack creates six S3 buckets. Every bucket uses S3-managed encryption,
blocks all public access, and enforces TLS.

| Purpose | Versioning | Removal policy | Lifecycle |
| --- | --- | --- | --- |
| Raw source data | Enabled | Retain | Transition objects to Standard-IA after 90 days |
| Curated data | Enabled | Retain | None |
| Knowledge artifacts | Enabled | Retain | None |
| Model artifacts | Enabled | Retain | None |
| Logs | Disabled | Destroy and auto-delete objects | Expire objects after 90 days |
| Athena results | Disabled | Destroy and auto-delete objects | Expire objects after 30 days |

The legacy `internal-dev` buckets keep CloudFormation-generated names. New
client environments use:

```text
{project}-{client}-{environment}-{layer}-{region}-{account}
```

The current default project value is `bah-de-assistant`. The knowledge layer
token is `kb`; the Athena-results token is `athena`. Changing these formulas
for a deployed client can replace buckets and is not a routine refactor.

Other resources are:

- One AWS Glue Data Catalog database. The legacy name is `dea_catalog`; new
  client environments use the client/environment resource prefix.
- One enforced Amazon Athena workgroup. The legacy name is `dea-workgroup`.
  Results are written under `query-results/` in the Athena-results bucket with
  SSE-S3 encryption.
- One Python 3.12 health-check Lambda function named `dea-health-check` for the
  legacy environment, with 256 MB memory and a 30-second timeout.
- One Python 3.12 document-ingestion Lambda named
  `dea-document-ingestion` for the legacy environment, with 512 MB memory, a
  five-minute timeout, reserved concurrency of one, and the pinned pure-Python
  `pypdf` parser bundled for local text-based PDF extraction.
- One `ObjectCreated` notification on the existing knowledge bucket, filtered
  only to `knowledge/raw/`, plus the S3-to-Lambda invoke permission.
- One SQS-managed encrypted dead-letter queue named
  `dea-document-ingestion-dlq` for failed asynchronous Lambda events. It
  requires TLS, retains messages for 14 days, and is deleted with the stack.
- The standard CDK auto-delete custom-resource provider used to empty the logs
  and Athena-results buckets when those disposable buckets are deleted. CDK
  synthesizes its helper Lambda and role; it is not an application endpoint.
- The standard CDK bucket-notification custom-resource provider used to apply
  the notification to the existing stack-owned bucket without a circular
  dependency. It receives `s3:PutBucketNotification` only for the knowledge
  bucket.
- One explicit Lambda log group at `/aws/lambda/dea-health-check` for the
  legacy environment, retained for one month and deleted with the stack.
- One explicit Lambda log group at
  `/aws/lambda/dea-document-ingestion`, also retained for one month and deleted
  with the stack.
- Eight SSM Parameter Store string parameters for the Glue database, Athena
  workgroup, and six bucket names. Legacy parameters are rooted at `/dea`;
  non-legacy parameters are rooted at the resource prefix.
- Separate Glue, health-check Lambda, and document-ingestion execution roles.
- CloudFormation outputs for the six bucket names, Glue database, Athena
  workgroup, Glue and health-check role ARNs, health function, client ID,
  environment, and resource prefix. The ingestion runtime adds no output, so
  the existing output contract is unchanged.

Resources receive the tags `Project=data-engineering-assistant`,
`ClientId={client}`, `Environment={environment}`, `ManagedBy=aws-cdk`, and
`Owner=bryan`. Client and environment scope are part of new-client resource
names and configuration.

## IAM boundaries

The Glue role can:

- list and locate only the raw and curated buckets;
- read raw objects and object versions;
- read and write curated objects;
- use the current account's Glue catalog and the stack's database tables and
  partitions; and
- write to standard AWS Glue log groups.

The health-check Lambda role can:

- list and read the knowledge bucket and object versions;
- read the eight SSM parameters created by the stack;
- invoke Bedrock foundation models in the stack Region; and
- write to standard Lambda log groups.

Neither role grants S3 bucket administration. The existing Bedrock permission
is Region-scoped but not model-ID-scoped; selecting and narrowing production
model access remains an explicit security decision. The local Streamlit
application does not run in the health-check Lambda and receives no credentials
from the stack.

The document-ingestion role is separate. It can:

- list only the `knowledge/metadata/` and `knowledge/embeddings/` prefixes so
  missing manifest or pending-descriptor state can be distinguished safely;
- read objects and versions only from `knowledge/raw/`;
- read manifest state from `knowledge/metadata/` and existing pending
  descriptors from `knowledge/embeddings/`;
- write only `knowledge/processed/`, `knowledge/chunks/`,
  `knowledge/embeddings/`, `knowledge/metadata/`, `knowledge/media/`, and
  `knowledge/quarantine/`;
- receive no read access to `knowledge/media/` or `knowledge/quarantine/`, so
  storage-only objects cannot enter automatic indexing;
- write only to its explicit log group; and
- send exhausted asynchronous failures to its dedicated queue.

It cannot write raw sources, access SSM, or invoke Bedrock. Its policy contains
no unrestricted allow resource and no `s3:*` action. Generated outputs cannot
retrigger the function because the notification has only the raw-prefix
filter. See
[Event-driven document ingestion](EVENT_DRIVEN_INGESTION.md).

PDF support changes only the function deployment asset and supported-format
configuration. It adds no AWS service, IAM action, secret, external parser
call, OCR runtime, or provider connection. CDK local bundling includes only the
handler, `knowledge`, `pypdf`, and the pinned `requests` dependency closure;
tests, virtual environments, caches, and native Windows extensions are not
packaged.

The CDK-generated auto-delete provider has its standard Lambda logging
permission and receives bucket-policy access to enumerate and delete objects in
only the two disposable buckets. That permission exists solely to implement
their configured `auto_delete_objects` behavior.

## Safe change workflow

Install the supported Python 3.12 dependencies, then synthesize locally:

```powershell
python -m pip install --constraint constraints.txt --requirement requirements.txt --requirement requirements-dev.txt
python -m pytest
cdk.cmd synth -c client=internal-dev
```

To compare with an existing deployed stack, use authenticated read access:

```powershell
cdk.cmd diff -c client=internal-dev
```

`cdk diff` is intentionally excluded from offline CI because it requires live
AWS context and can create a read-only CloudFormation change set. Review every
replacement, IAM change, physical name, removal policy, and output before
considering deployment.

Deployment is a manual operator action and may create costs:

```powershell
cdk.cmd deploy -c client=internal-dev --require-approval broadening
```

Confirm the intended AWS account and Region, review the diff and generated
change set, and obtain the required project approval before running it. This
repository's CI never deploys.

CloudFormation normally rolls back a failed create or update automatically.
Inspect events before intervening:

```powershell
aws cloudformation describe-stack-events --stack-name DataEngineeringAssistantCdkStack --region us-west-2
```

If CloudFormation reports `UPDATE_ROLLBACK_FAILED`, investigate the failed
resource first. Continuing rollback is an operator-controlled recovery action:

```powershell
aws cloudformation continue-update-rollback --stack-name DataEngineeringAssistantCdkStack --region us-west-2
```

Do not skip resources or force recovery without an incident-specific plan.

## Destruction and retained data

Destroying a stack is destructive and requires explicit human confirmation:

```powershell
cdk.cmd destroy -c client=internal-dev
```

Before confirming, verify the AWS account, Region, stack ID, backups, and data
owners. The raw, curated, knowledge, and model buckets use `RETAIN`, so they
remain after stack deletion and may continue to incur storage costs. The logs
and Athena-results buckets, their contents, both application Lambda log groups,
the ingestion failure queue, and other non-retained resources are configured
for deletion.

Never use destroy as a rollback technique, and never automate this command in
CI.

## Cost boundaries

S3 storage and requests, Glue catalog or job usage, Athena scanned data,
CloudWatch Logs, Lambda invocations, SQS failure storage, SSM API usage, and
Bedrock inference can all incur charges. A supported upload under
`knowledge/raw/` can invoke Lambda after deployment. The stack does not create
Glue jobs, scheduled ingestion, Athena queries, a managed vector database,
alarms, dashboards, or a hosted Streamlit service. Application cost estimates
cover supported LLM requests only and are not AWS billing records.
## Production indexing foundation

Automatic production indexing is opt-in. When enabled with reviewed context,
the existing ingestion Lambda receives an external Python 3.12 dependency
layer, exact Bedrock model and Secrets Manager permissions, optional exact KMS
decrypt permission, and optional attachment to imported private networking.
No VPC, NAT gateway, Qdrant service, collection, or secret is created. See
[Production vector indexing runtime](PRODUCTION_VECTOR_INDEXING_RUNTIME.md).
