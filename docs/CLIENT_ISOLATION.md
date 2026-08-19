# Client isolation foundation

## Current approach

This repository uses one CDK codebase and selects exactly one client
configuration for each synth or deployment. Each client/environment combination
gets a separate stack identity, client-aware names for newly introduced
deployments, and `ClientId` and `Environment` tags.

The existing `internal-dev` configuration is a compatibility case. Its current
physical resource naming is preserved because adding explicit names to deployed
resources, especially S3 buckets, can cause replacement. Other client
configurations use the normalized resource prefix:

`{project}-{clientId}-{environment}`

This is a foundation, not complete tenant isolation. Client data must never be
mixed across prefixes, buckets, Glue databases, logs, or secrets.

The `internal-dev` configuration retains the legacy
`DataEngineeringAssistantCdkStack` identity so an existing deployment remains
the update target. All non-legacy configurations use client-specific stack
identities, such as
`DataEngineeringAssistant-Demo-Client-Dev-Stack`.

## Future isolation

Later phases must add:

- Separate IAM roles and secrets for every client.
- Separate workflow monitoring, logs, and alarms for every client.
- Separate AWS accounts for larger or regulated clients.
- Cross-account deployment.
- Automated client onboarding.
- Billing and SLA automation.

AWS Organizations, cross-account deployment, automated onboarding, billing,
and SLA automation are intentionally deferred from the current implementation.
