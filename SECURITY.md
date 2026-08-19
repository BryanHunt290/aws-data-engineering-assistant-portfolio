# Security policy

## Supported versions

Security fixes are applied to the latest release on the default branch. This
project does not currently maintain multiple supported release lines.

## Reporting a vulnerability

Before the first public release, the repository owner should enable **Private
vulnerability reporting** under the GitHub repository's Security settings.
After it is enabled, use **Security > Advisories > Report a vulnerability** so
details remain private.

If private reporting is not available, open a public issue containing only a
request for a private reporting channel. Do not include exploit details,
credentials, account identifiers, customer data, or other sensitive material
in a public issue.

Include the affected version or commit, impact, reproduction conditions, and a
minimal proof of concept with secrets and real data removed. Please allow the
maintainer time to validate and coordinate a fix before public disclosure.

## Secrets and data

- Never commit AWS credentials, access tokens, `.env` files, account IDs, or
  customer documents.
- Use the synthetic CC0 demo corpus for examples, tests, screenshots, and issue
  reports.
- Supply boto3 credentials through the standard external credential chain.
  Prefer short-lived, least-privilege credentials.
- Treat exported session feedback as potentially sensitive and review it
  before sharing. Feedback and conversations are session-local by default.
- The application does not ask for or display AWS credentials.

## Application and container security

The default Streamlit and Docker mode is offline and uses deterministic fake
providers. Bedrock mode is opt-in, makes billable AWS API calls, and must be
given access only to the selected embedding and generation models. The Docker
image runs as a non-root user and does not contain credentials. Authentication,
TLS termination, persistent multi-user storage, and hosted-service hardening
are not implemented; do not expose the local application directly to the
public internet.

## Infrastructure security

AWS CDK commands can affect billable resources. Review `cdk diff` and the
CloudFormation change set before deployment. Never run deployment or
destruction from untrusted contributions or CI. Preserve the legacy
`DataEngineeringAssistantCdkStack` identity for `internal-dev`; changing stack
or physical resource identities can create replacements or duplicate
infrastructure.

The stack blocks public S3 access, enforces TLS, uses S3-managed encryption, and
scopes application roles to the implemented resources. These controls do not
replace an organization-level security review, account guardrails, logging,
backup, incident response, or cost controls.
