# Synthetic IAM Least-Privilege Guide

License: CC0-1.0. This is synthetic demonstration content.

Begin with the workload's required API calls and exact resource boundaries.
Separate the role trust policy from its permission policies. Scope S3 access to
the necessary buckets and prefixes, Glue access to the relevant catalog,
database, and tables, and CloudWatch Logs access to the intended log groups.

Use access-denied evidence and service authorization references to refine a
policy. Avoid wildcard actions and resources unless the AWS service requires
them and the exception is documented. Conditions can constrain Regions,
resource tags, encryption, transport security, or calling services.

Review changes before applying them. A policy recommendation is not approval
to broaden production permissions.
