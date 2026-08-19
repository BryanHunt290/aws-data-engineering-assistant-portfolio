# Synthetic S3 Data Lake Architecture Guide

License: CC0-1.0. This is synthetic demonstration content.

Use separate S3 prefixes or buckets for landing, validated, curated, and query
result data. Enable encryption, block public access, and use versioning where
recovery requirements justify it. Lifecycle rules should reflect retention and
access patterns rather than being copied between workloads.

An ingestion pipeline can land immutable source objects in S3, run validation,
transform data with AWS Glue, register curated tables in the Glue Data Catalog,
and query those tables through a controlled Athena workgroup. Partition keys
should match common filters without creating excessive small partitions.

Treat bucket names, database names, workgroups, Regions, and schedules as
requirements to confirm. A design recommendation is not evidence that any AWS
resource already exists.
