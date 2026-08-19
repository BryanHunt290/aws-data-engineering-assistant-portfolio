# Synthetic Athena Query Troubleshooting Guide

License: CC0-1.0. This is synthetic demonstration content.

For an Athena query failure, preserve the query execution ID and inspect the
reported error category. Confirm the selected workgroup, output location,
database, table definition, and caller permissions. The caller needs access to
source data and the query-results location; encrypted data may also require KMS
permissions.

Schema mismatches often appear when file types or column types differ across
partitions. Inspect representative objects and repair catalog metadata only
after confirming the actual storage format. Partition projection or partition
repair should match the physical S3 layout.

For slow or expensive queries, scan fewer columns, filter on partition keys,
compact small files, and prefer columnar formats such as Parquet.
