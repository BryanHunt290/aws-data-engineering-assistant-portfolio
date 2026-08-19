# Synthetic Glue Job Troubleshooting Guide

License: CC0-1.0. This is synthetic demonstration content.

Start with the first meaningful CloudWatch error and the denied AWS API, ARN,
or data path. For AccessDenied failures, verify the Glue job execution role,
its trust policy, identity policies, S3 bucket policies, and any KMS key policy.
Confirm that the role can list the bucket and access the exact object prefix.

Separate permission failures from missing objects, schema problems, dependency
errors, exhausted workers, and network connectivity. Reproduce with the
smallest safe input, preserve the failed run identifier, and compare job
arguments with the last successful run.

Do not broaden permissions to `*` as a first response. Add only the confirmed
actions and resources, then rerun a non-production test before promotion.
