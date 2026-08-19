# Non-production vector indexing validation

This runbook prepares the first controlled production-vector-indexing test in
the existing `internal-dev` environment. Preparation is offline and stops
before an authenticated `cdk diff`, deployment, provider connection, model
invocation, secret access, layer publication, fixture upload, or cleanup.

The committed example uses a reserved `.invalid` endpoint and synthetic
zero-account ARNs. The preparation command copies those placeholders into a
Git-ignored local file; it never creates a credential or populates a secret.
The existing runtime remains disabled in `cdk.json` unless both
`automaticIndexingEnabled=true` and `integrationValidationEnabled=true` are
present in reviewed context. The integration gate rejects any scope other than
`internal-dev`.

## Preparation flow

```text
committed example
  -> ignored local configuration
  -> field-by-field offline review
  -> existing offline preflight
  -> reviewed Lambda layer archive
  -> generated PowerShell CDK context + provenance audit
  -> cdk synth --no-lookups
  -> synthesized-template resource review
  -> STOP for explicit approval
```

The field review reports only field names, categories, statuses, and safe
messages. Its statuses are `ready`, `placeholder`, `missing`, `invalid`,
`optional-not-configured`, and `prohibited`. It does not print configuration
values. The existing preflight remains available and prints only safe
identifiers, named checks, and counts.

## Stage A — repository review

Run from a PowerShell session with Python 3.12. These commands are local and
make no network or provider call:

```powershell
Set-Location C:\path\to\data-engineering-assistant-cdk
python --version
git status --short
git diff --stat
git diff --check
git check-ignore --no-index `
  .\config\integration-validation.internal-dev.local.json `
  .\.local\internal-dev-indexing-context.ps1 `
  .\.local\internal-dev-indexing-audit.json
```

Expected: Python reports `3.12.x`; all three generated paths are ignored. Do
not discard or overwrite existing working-tree changes.

## Stage B — create ignored local configuration

```powershell
python -m scripts.prepare_indexing_integration bootstrap
```

The command creates:

```text
config/integration-validation.internal-dev.local.json
```

It refuses an existing file. Only after inspecting that exact file and deciding
to replace it may an operator use the bounded override:

```powershell
python -m scripts.prepare_indexing_integration bootstrap --force
```

The output separates required non-secret fields, optional VPC fields, secret
reference fields, and prohibited plaintext API-key field names. It displays no
secret value. A custom output is accepted only inside the repository and only
when it matches an ignored `config/*.local.json`, `.local/`, or `build/` path.

## Stage C — populate reviewed non-secret identifiers

```powershell
notepad.exe .\config\integration-validation.internal-dev.local.json
```

Replace the reserved endpoint and synthetic ARNs with reviewed non-secret
identifiers. Confirm the following without adding credential material:

- `client=internal-dev`, `clientId=internal`, and `environment=dev`;
- both opt-in flags are true;
- `awsRegion` matches the Bedrock, secret, layer, and optional KMS ARN Regions;
- the model or inference-profile ARN is approved for internal development;
- the Qdrant endpoint is credential-free HTTPS and its collection includes
  `internal_dev`;
- `indexingQdrantSecretArn` is an ARN reference only—the referenced secret must
  contain the API key, not this file;
- the versioned dependency-layer ARN and embedding dimensions match the layer,
  model, and collection contract;
- namespace, domain, timeouts, retry counts, descriptor batches, and chunk
  limits are intentionally bounded;
- VPC ID, private subnet IDs, availability zones, and Qdrant security-group ID
  are either all reviewed and complete or all left unconfigured.

The knowledge bucket reference is derived from the stack-managed
`KnowledgeBucket`; no bucket name or ARN belongs in local context. Never add
`appQdrantApiKey`, `indexingQdrantApiKey`, `knowledgeQdrantApiKey`, or
`qdrantApiKey`, even with an empty value.

### Qdrant Cloud field matrix

This integration uses the public Qdrant Cloud database endpoint. It does not use
the Qdrant Cloud management API and does not provision a Qdrant cluster. The
reviewed non-production configuration requires:

- `indexingQdrantEndpointSource=environment`;
- the credential-free HTTPS cluster URL in `indexingQdrantUrl`;
- the scoped `internal_dev` collection name;
- a Secrets Manager ARN in `indexingQdrantSecretArn` and no plaintext key;
- `indexingTlsRequired=true`; and
- a versioned `indexingDependencyLayerArn` containing the pinned
  `qdrant-client` dependency.

For a public Qdrant Cloud endpoint, these values may all remain empty:

- `indexingVpcId`;
- `indexingSubnetIds`;
- `indexingAvailabilityZones`; and
- `indexingQdrantSecurityGroupId`.

With all four empty, Lambda remains outside a customer VPC and uses its default
public internet access. Supplying any one selects the existing private-routing
mode and requires the complete VPC set; that mode remains intended for a
security-group-reachable private or self-hosted Qdrant service. Do not put a
Qdrant Cloud hostname into a partial VPC configuration.

`indexingQdrantKmsKeyArn` may remain empty only when the referenced secret uses
the AWS-managed Secrets Manager KMS key. If it uses a customer-managed key, the
key ARN is required so the Lambda role receives scoped `kms:Decrypt` permission.

The dependency-layer ARN is not a Qdrant Cloud networking field and is not
optional for a deployable indexing Lambda. The base Lambda asset excludes
`qdrant-client`; the versioned layer supplies it. Operators may populate the
Qdrant Cloud endpoint and secret reference before the layer is published, but
field review must remain `NOT READY` and context generation must remain blocked
until the reviewed versioned layer ARN is present.

Qdrant Cloud Database API keys should be short-lived and least-privileged. If a
key is collection-scoped, pre-create the reviewed collection; automatic
creation needs sufficient database permissions. Qdrant client-IP restrictions
also require a separately reviewed stable-egress design, which this public
non-VPC mode does not provide. See the
[external Qdrant integration contract](QDRANT_INTEGRATION_CONTRACT.md).

## Stage D — offline field review

PowerShell-friendly human output:

```powershell
python -m scripts.prepare_indexing_integration review `
  --config .\config\integration-validation.internal-dev.local.json
if ($LASTEXITCODE -ne 0) { throw "Local integration configuration is not ready" }
```

Secret-safe JSON for later automation:

```powershell
python -m scripts.prepare_indexing_integration review `
  --config .\config\integration-validation.internal-dev.local.json `
  --format json
if ($LASTEXITCODE -ne 0) { throw "Local integration configuration is not ready" }
```

Exit code `0` means every required field and the existing preflight are ready.
Exit code `2` means preparation must stop. `optional-not-configured` is not a
failure when every VPC field is absent. The example configuration intentionally
exits `2` because it retains endpoint and ARN placeholders.

## Stage E — offline preflight

Run the existing validation command after the field review succeeds:

```powershell
python -m scripts.validate_indexing_integration `
  --config .\config\integration-validation.internal-dev.local.json
if ($LASTEXITCODE -ne 0) { throw "Integration preflight failed" }
```

The preflight checks Python 3.12, explicit scope, both gates, existing client
configuration, provider types, ARN syntax, HTTPS, collection scope, dimensions,
namespace/domain, timeouts/retries, work limits, optional KMS, VPC completeness,
and absence of plaintext API-key fields. It also emits the seven dry-run
phases: `preflight`, `readiness`, `upload_plan`, `indexing_observation`,
`verification`, `failure_path_validation`, and `cleanup_plan`. It makes zero
network calls and has no live executor.

## Stage F — build and inspect the Lambda layer

Building downloads pinned Python wheels from the configured package index. It
is a local build—not an AWS operation—but requires outbound package-network
access. Skip the build until that access is allowed. It never publishes a layer:

```powershell
.\scripts\build_indexing_runtime_layer.ps1
python -m scripts.inspect_indexing_runtime_layer `
  --archive .\build\indexing-runtime-layer.zip `
  --requirements .\lambda\indexing_runtime_requirements.txt
Get-FileHash .\build\indexing-runtime-layer.zip -Algorithm SHA256
Get-Item .\build\indexing-runtime-layer.zip | Select-Object Name,Length
```

The build requires Python 3.12, installs pinned manylinux wheels, refuses output
outside `build/`, and refuses overwrite. The inspector checks the `python/`
root, pinned `qdrant-client` metadata, archive size and SHA-256, file count, and
native/platform files. `-Clean` removes only the selected local build directory
and ZIP; use it only after reviewing the resolved target:

The builder installs the fully pinned transitive dependency closure from
`lambda/indexing_runtime_requirements.lock.txt` with `--no-deps`. This keeps a
Windows build host from evaluating Windows-only dependency markers while pip is
selecting CPython 3.12 manylinux2014 x86_64 wheels.

```powershell
.\scripts\build_indexing_runtime_layer.ps1 -Clean
```

A versioned Lambda layer ARN must come from an existing approved non-production
layer or a separately approved future publication. Publication is not part of
this preparation milestone.

## Stage G — generate CDK context and provenance

```powershell
python -m scripts.prepare_indexing_integration generate-context `
  --config .\config\integration-validation.internal-dev.local.json
if ($LASTEXITCODE -ne 0) { throw "Context generation failed" }

. .\.local\internal-dev-indexing-context.ps1
```

The command refuses a failed review, output outside `.local/` or `build/`, and
existing output unless `--force` is explicitly supplied. It normalizes booleans
to lowercase and arrays to comma-separated CDK values, then writes the exact
`-c key=value` argument array. The generated script is labeled safe only for
`cdk synth --no-lookups`, not approved for `cdk diff` or `cdk deploy`.

It also writes `.local/internal-dev-indexing-audit.json` with:

- UTC timestamp, commit hash when available, and dirty/clean state;
- Python version;
- SHA-256 of canonical normalized non-secret configuration;
- local config and generated context paths;
- the complete safe preflight result;
- expected collection, client, environment, namespace, domain, embedding
  dimensions, and stack name.

The artifact contains no environment dump or secret value. Repeated generation
with identical configuration produces the same fingerprint; timestamp and
repository state are expected to vary.

## Stage H — run no-lookup synthesis

This is the only approved use of the generated context in this milestone:

```powershell
cdk.cmd synth @ContextArgs --no-lookups --quiet `
  --output .\.local\cdk.out.internal-dev-integration
if ($LASTEXITCODE -ne 0) { throw "Offline integration synth failed" }
```

For regression comparison, synthesize the two default-disabled client scopes
without the generated opt-in context:

```powershell
cdk.cmd synth -c client=internal-dev --no-lookups --quiet `
  --output .\.local\cdk.out.internal-default
cdk.cmd synth -c client=demo-client-dev --no-lookups --quiet `
  --output .\.local\cdk.out.demo-client-dev
```

`--no-lookups` is mandatory. Do not run an authenticated diff.

## Stage I — inspect synthesized resources

Generate the deterministic expectation report and assert the actual synthesized
template:

```powershell
python -m scripts.prepare_indexing_integration expected-resources `
  --config .\config\integration-validation.internal-dev.local.json `
  --template .\.local\cdk.out.internal-dev-integration\DataEngineeringAssistantCdkStack.template.json
if ($LASTEXITCODE -ne 0) { throw "Synthesized resource review failed" }

Get-Content .\.local\internal-dev-expected-resources.json -Raw
```

The report checks:

- document-ingestion Lambda runtime, optional reserved concurrency matching the
  reviewed configuration, dependency layer, complete bounded indexing
  environment, and stack-managed bucket ref;
- model-scoped Bedrock permission, secret-ARN-scoped Secrets Manager permission,
  and optional KMS-key-scoped decrypt permission;
- optional Lambda VPC attachment, the reviewed VPC/subnets, Secrets Manager,
  Logs, Bedrock Runtime, and S3 interface endpoints, separate Lambda/endpoint
  security groups, and TCP/443 egress to the reviewed Qdrant security group;
- encrypted 14-day DLQ attachment and raw-prefix-only S3 ObjectCreated event;
- absence of plaintext Qdrant credentials, newly created Qdrant infrastructure,
  and production-client resource leakage.

The output is deterministic and contains the explicit labels `safe_for: cdk
synth --no-lookups` and `not_approved_for: cdk diff, cdk deploy`.

## Stage J — stop and request approval

Stop here. Do not authenticate CDK, deploy, connect to providers, access a
secret, publish a layer, or upload the fixture. Preserve the local review,
context, audit, synthesized template, and expected-resource report for review.

The next action requiring explicit approval is an authenticated
`cdk diff @ContextArgs`. Approval must identify the AWS account, Region,
internal-dev stack, reviewed layer version, Qdrant collection, network route,
cost owner, rollback plan, and evidence-retention plan. Approval for a diff does
not grant approval to deploy.

## Later action classification

| Classification | Later examples | Current status |
| --- | --- | --- |
| Local/offline | review, preflight, archive inspection, `synth --no-lookups`, template review | Allowed in stages A–I |
| External package read | download pinned layer wheels | Separate package-network approval if required |
| Read-only AWS | authenticated `cdk diff`, stack/output/log inspection, resource metadata checks | Stop; explicit approval required |
| Billable provider operation | Bedrock embedding, Qdrant query/upsert, deployed Lambda execution | Stop; explicit approval required |
| State-changing | layer publication, secret create/update, `cdk deploy`, fixture upload, failure injection, redrive apply | Stop; separate explicit approval required |
| Destructive | `cdk destroy`, object/collection/secret/layer deletion, rollback that removes retained evidence | Prohibited without target-specific destructive approval |

Reading a Secrets Manager value is intentionally excluded even from the
read-only preparation category: this milestone neither creates nor reads a
secret. Qdrant readiness and retrieval are also live provider operations even
when they do not mutate vectors.

## Future controlled validation boundary

After separately approved deployment, the repository fixture is synthetic and
expected to create four chunks using the documented chunk settings. Evidence
verification already checks schema-v2 descriptor scope, manifest state,
collection and vector dimensions, client/environment/namespace/domain payload
scope, touched prefixes and collections, DLQ behavior, permanent-failure
redrive rejection, and sensitive-log exclusions. The runner still contains no
live executor, and cleanup remains manual-only.

Known limitations: offline review cannot prove provider reachability, model
authorization, secret schema, published-layer contents, Qdrant collection
distance/dimensions, deployed event delivery, or retrieval quality. The runtime
cannot provide an atomic transaction across S3, Bedrock, Qdrant, and the
manifest. Those require later, explicitly approved live evidence collection.
