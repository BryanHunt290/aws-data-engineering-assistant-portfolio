# External Qdrant integration contract

Qdrant is externally provisioned. The CDK stack creates no Qdrant cluster,
service, load balancer, collection, backup, or credential.

## Endpoint, TLS, and authentication

- Production and integration endpoints must be absolute `https://` URLs.
- URL credentials, query strings, and fragments are rejected.
- Authentication is mandatory. The Lambda environment contains only a Secrets
  Manager ARN, never an API key.
- The referenced `SecretString` must be a JSON object with exactly a non-empty
  string `api_key` and, only when endpoint source is `secret`, an optional
  HTTPS `endpoint`. Binary secrets and extra fields fail closed.
- Secret retrieval is lazy and cached. Rotation is visible to new Lambda
  execution environments; a warm environment may retain the old value until
  recycled. Coordinate overlapping credential validity during rotation.

For Qdrant Cloud, use the database cluster endpoint shown on the cluster detail
page, not the Qdrant Cloud management API endpoint. The Lambda uses the Qdrant
database API and a Database API key. A Cloud Management key is not a substitute.
The repository configuration keeps the endpoint in `indexingQdrantUrl`, sets
`indexingQdrantEndpointSource` to `environment`, and stores only the Database
API key in the referenced Secrets Manager secret.

Exact `SecretString` shape (`endpoint` is optional):

```json
{
  "api_key": "<value-managed-outside-source-control>",
  "endpoint": "https://qdrant.integration.invalid"
}
```

## Managed Qdrant Cloud over a public HTTPS endpoint

The supported Qdrant Cloud mode uses the externally managed public cluster
endpoint over HTTPS. The CDK stack neither provisions nor administers the
Qdrant Cloud account, cluster, API key, collection, backups, or allowlist.

Required Qdrant and runtime fields before an authenticated diff or deployment:

| Field | Requirement |
| --- | --- |
| `indexingVectorStoreProvider` | Must be `qdrant`. |
| `indexingQdrantEndpointSource` | Must be `environment` for offline-reviewable Qdrant Cloud configuration. |
| `indexingQdrantUrl` | Credential-free absolute HTTPS cluster URL; query strings and fragments are rejected. |
| `indexingQdrantCollection` | Lowercase, scoped collection containing `internal_dev`. |
| `indexingQdrantSecretArn` | Complete Secrets Manager ARN only; never the Database API key. |
| `indexingTlsRequired` | Must be `true`. |
| `indexingDependencyLayerArn` | Versioned Lambda layer ARN containing the pinned `qdrant-client`; required before diff/deployment readiness. |

The following fields may remain empty for the public Qdrant Cloud mode:

| Field | Empty-field behavior |
| --- | --- |
| `indexingVpcId` | Lambda remains outside a customer VPC and retains default public internet access. |
| `indexingSubnetIds` | No private subnets are imported. |
| `indexingAvailabilityZones` | No subnet-to-availability-zone mapping is needed. |
| `indexingQdrantSecurityGroupId` | A public Qdrant Cloud endpoint has no AWS security group to import. |
| `indexingQdrantKmsKeyArn` | Valid when the secret uses the AWS-managed Secrets Manager KMS key; provide the customer-managed key ARN otherwise. |

All four VPC fields are a single private-routing mode. Leaving all four empty is
valid. Supplying any one requires a complete, internally consistent VPC,
subnet, availability-zone, and Qdrant security-group set. That existing mode is
for a security-group-reachable private Qdrant service; it is not required for a
public Qdrant Cloud cluster and is unchanged for self-hosted deployments.

Qdrant Cloud can restrict cluster access by client IP. A Lambda outside a
customer VPC does not provide a repository-controlled static egress IP. The
current private-routing mode also does not model NAT/static-IP egress to a
public Qdrant Cloud endpoint. If client IP restrictions are mandatory, stable
egress design is a separate networking milestone and must be approved before
deployment.

The Database API key should use the narrowest permissions and expiration that
support the integration. A collection-scoped key requires the collection to be
pre-created. Allowing the adapter to create a missing collection requires an
appropriately authorized key. The integration never uses a Cloud Management
key and never writes an API key into CDK context, Lambda environment variables,
logs, audit artifacts, or source control.

## Collection contract

The reviewed collection name must use lowercase letters, numbers, and
underscores, be at most 120 characters, and include the normalized
client/environment token—for the fixture, `internal_dev`. Every collection uses one unnamed dense-vector
configuration with the explicitly configured embedding dimension and cosine
distance.

On first upsert, `QdrantVectorStore` creates a missing collection using those
settings. Operators may instead pre-create it. If an existing collection uses
another dimension, named vectors, malformed configuration, or a distance other
than cosine, indexing stops with a permanent configuration error; it never
recreates or modifies that collection automatically.

Required scope payload fields are `client_id`, `environment`, `namespace`,
`knowledge_namespace`, `domain`, and `knowledge_domain`. Deterministic point IDs
include client, environment, namespace, domain, document, chunk, checksum, and
embedding model. The remaining source and timestamp fields support diagnostics.
Retrieval always applies database-side `client_id` and `environment` filters;
namespace/domain filters should also be supplied for this validation. Evidence
collection must report payload identifiers/counts only—not stored text or
vectors.

## Availability, retry, and private routing assumptions

The integration example uses 5-second connect configuration, 15-second request
configuration, and two wrapper retries. Qdrant's Python client receives the
request timeout; lower-level connection behavior remains transport-dependent.
Authentication, TLS, and incompatible-collection failures need operator action
or are permanent. Bounded unavailability is retryable and leaves chunks
pending. Retries are not an availability SLA.

For private Qdrant, its supplied security group must be reachable on TCP/443
from the imported Lambda subnets. Private DNS, certificates, routes, endpoint
policies, and availability across those subnets are external prerequisites.
The optional AWS interface endpoints do not provide routing to Qdrant itself.

References:

- [Qdrant Cloud database authentication](https://qdrant.tech/documentation/cloud/authentication/)
- [Qdrant Cloud cluster access](https://qdrant.tech/documentation/cloud/cluster-access/)
- [AWS Lambda VPC and internet access](https://docs.aws.amazon.com/lambda/latest/dg/configuration-vpc.html)
- [AWS Secrets Manager KMS permissions](https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html)
