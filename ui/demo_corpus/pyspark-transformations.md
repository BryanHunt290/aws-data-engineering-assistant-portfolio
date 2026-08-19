# Synthetic PySpark Transformation Patterns

License: CC0-1.0. This is synthetic demonstration content.

For deterministic deduplication, partition a window by the business key and
order by a reliable update timestamp plus a stable tie-breaker. Keep the first
row number and drop the temporary rank column. Define how null keys and equal
timestamps should behave.

Prefer built-in Spark SQL functions over Python UDFs when possible. Validate
input schemas, select only required columns, and avoid collecting production
datasets to the driver. Repartition based on measured data distribution rather
than a fixed rule.

Test transformations with duplicates, nulls, late-arriving updates, malformed
records, and empty input. Record assumptions alongside generated code.
