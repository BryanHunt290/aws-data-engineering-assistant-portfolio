# Production vector indexing runtime

## Architecture and trust boundaries

S3 ObjectCreated invokes the existing Python 3.12 ingestion Lambda. Extraction
and chunk persistence finish before the schema-v2 descriptor is written; the
existing `VectorIndexingWorkflow` then composes `EmbeddingWorkflow`, lazy
Bedrock embeddings, and a lazy authenticated `QdrantVectorStore`. S3 is the
durable descriptor/manifest boundary. Secrets Manager is the credential trust
boundary. Qdrant remains an external, customer-selected service: this stack
does not invent an ECS or hosted-Qdrant architecture and never creates a
collection during synthesis or cold start.

Automatic indexing is disabled by default because enabling it creates an
external data path, requires reviewed model/vector-store contracts, and may
incur Lambda, Bedrock, Secrets Manager, VPC endpoint, log, S3, SQS, KMS, and
external Qdrant costs.

## Configuration

Runtime variables contain only non-secret values and references. Production
requires `KNOWLEDGE_INDEXING_RUNTIME_MODE=production`, enabled indexing,
`bedrock`, `qdrant`, client/environment/namespace/domain, model ID, collection,
secret identifier, endpoint source, TLS/authentication flags, timeouts, retry
limits, descriptor batch limit, and chunks-per-invocation limit. Invalid enum,
scope, numeric, authentication, or endpoint combinations raise typed
`IndexingConfigurationError` before provider work.

CDK context is opt-in with `automaticIndexingEnabled=true`. Required fields are
`indexingBedrockModelArn`, `indexingQdrantCollection`,
`indexingQdrantSecretArn`, and `indexingDependencyLayerArn`. An environment
endpoint additionally requires `indexingQdrantUrl`. Optional fields are
`indexingQdrantKmsKeyArn`, `indexingReservedConcurrentExecutions`,
`indexingVpcId`, comma-separated
`indexingSubnetIds`, matching `indexingAvailabilityZones`, and
`indexingQdrantSecurityGroupId`. Plaintext API-key context is rejected. Values
must be client/environment-specific; do not reuse a collection or secret across
tenants.

## Secrets, authentication, and TLS

The referenced Secrets Manager `SecretString` must be JSON with a required
`api_key` string and optional HTTPS `endpoint` string. The endpoint is used only
when its configured source is `secret`. Retrieval is lazy and cached for the
Lambda execution environment. Values are never logged, returned in exceptions,
placed in templates, or output by CloudFormation. Production rejects plaintext
credentials and non-HTTPS endpoints. HTTP is accepted only for explicit local
loopback use.

## Composition, readiness, and safe observability

Providers are constructed through the existing composition function and remain
injectable. Bedrock creates its SDK client only on first embedding work; the
secret and Qdrant client are resolved only on first store operation.
`check_indexing_readiness` validates settings and optional composition without
connectivity calls. No live probe runs at import, synth, or test time.

Structured logs contain client, environment, namespace, domain, document ID,
provider/store type, status, counts, retry count, safe failure stage/type, and
duration. They contain no text, prompt, vector, credential, endpoint, or raw
provider exception. High-cardinality identifiers belong in logs, not metric
dimensions.

## Dependency packaging

Production attaches an explicitly supplied Python 3.12 Lambda layer ARN. The
direct runtime dependency is pinned in `lambda/indexing_runtime_requirements.txt`,
and its reviewed transitive closure is pinned in
`lambda/indexing_runtime_requirements.lock.txt`.
Build it from Windows PowerShell:

```powershell
.\scripts\build_indexing_runtime_layer.ps1
```

The script installs the complete lock with `--no-deps`, asks pip only for
manylinux2014 x86_64 CPython 3.12 wheels, and packages only runtime
dependencies. Avoiding dependency resolution during the build also prevents
Windows-only environment markers from entering the Linux layer. It needs
package-index access but not Docker. Unit
tests and synth do not build the layer and need neither Docker nor Qdrant.
Publishing the ZIP and supplying its versioned layer ARN are deployment
prerequisites.

## Networking and IAM

Networking is absent by default. When an existing VPC, private subnets, AZs,
and Qdrant security group are supplied, Lambda uses a new deny-by-default
security group with TCP/443 egress only to Qdrant and endpoint security groups.
Interface endpoints for S3, CloudWatch Logs, Secrets Manager, and Bedrock Runtime
avoid a NAT gateway; they cost money. No CIDR is hard-coded and no VPC/NAT is
created. Private DNS must be enabled on the existing VPC.

When enabled, the ingestion role receives `bedrock:InvokeModel` for the exact
configured model/inference-profile ARN and `secretsmanager:GetSecretValue` for
the exact secret ARN. Optional customer KMS encryption adds `kms:Decrypt` for
only that key. Disabled indexing receives none of these grants. VPC attachment
requires AWS-managed ENI actions whose resource type cannot be ARN-scoped; CDK
adds those only for VPC-enabled Lambda.

The ingestion role reads only the raw source plus the persisted chunks,
metadata, and embedding records required by the workflow. Its bounded
`s3:GetObject` resources include `knowledge/chunks/*`, while `s3:ListBucket`
remains limited to the metadata and embeddings prefixes used for conditional
manifest and missing-record handling.

Reserved concurrency is opt-in. When
`indexingReservedConcurrentExecutions` is absent or empty, the Lambda uses the
account's unreserved concurrency pool. An explicitly configured value must be
a positive integer; zero is rejected. Internal-dev leaves this field unset so
its ten-unit account quota retains AWS's required ten unreserved units.

## Manifest consistency, retry, and DLQ

The aggregate S3 manifest uses ETag optimistic locking: read object plus ETag,
write with `If-Match` (or `If-None-Match` when creating), detect conflict,
re-read, reapply the single-entry mutation, and retry to a bounded limit.
Exhaustion raises `ManifestWriteConflictError`. This preserves concurrent
documents and compatible legacy fields; descriptor-first persistence and
per-chunk indexed state are unchanged. S3 conditional updates do not provide a
multi-object transaction, so a process can still stop between descriptor and
manifest writes; idempotent event retry reconciles that window. Exhausted
asynchronous Lambda retries reach the existing encrypted 14-day DLQ.

## Redrive

The provider-neutral service inspects only manifest-referenced descriptors,
requires exact client/environment scope, supports namespace/domain/document/
status filters, and reports identifiers/counts only. Dry-run is the default.
Permanent validation failures are never automatically reset or dispatched.

```powershell
python .\scripts\redrive_indexing.py --bucket <bucket-name> --client-id <client> --environment prod --status failed
python .\scripts\redrive_indexing.py --bucket <bucket-name> --client-id <client> --environment prod --document-id <id> --apply --reset-retryable --function-name <lambda-name>
```

The second command mutates state and invokes AWS; review the dry run first.
Unit tests inject storage and dispatch fakes and make no AWS/provider/store
calls.

## Deployment prerequisites, rollback, and limitations

Before enabling, create and populate a client-specific secret outside this
repository, provision/review durable Qdrant and its collection contract, publish
the layer, authorize the model, validate private DNS/routes when applicable,
and stage a dry-run redrive. Roll back by disabling indexing while leaving
descriptors and the DLQ intact; do not delete the collection or secret until
retention requirements are reviewed.

The stack does not host Qdrant, rotate credentials, run connectivity probes,
or guarantee atomicity across descriptor, vector store, and manifest. Interface
endpoints and the external vector service are operational/cost dependencies.

Implementation readiness does not constitute live integration evidence. Use
the separately gated [non-production validation runbook](NON_PRODUCTION_VECTOR_INDEXING_VALIDATION.md)
and [Qdrant integration contract](QDRANT_INTEGRATION_CONTRACT.md). No deployment,
connectivity probe, failure injection, or cleanup is authorized merely by the
presence of this runtime.
