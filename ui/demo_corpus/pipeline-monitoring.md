# Synthetic Pipeline Monitoring Guide

License: CC0-1.0. This is synthetic demonstration content.

Monitor pipeline freshness, completion, duration, processed-record counts,
rejected-record counts, and data-quality outcomes. Track infrastructure signals
such as Glue failures, throttling, worker exhaustion, Lambda errors, and Athena
query failures without treating a single metric as proof of root cause.

Alarms should identify the client, environment, pipeline, and run while
avoiding sensitive payloads. Define owners, escalation paths, retry behavior,
and runbooks before production. Distinguish a delayed upstream feed from an
execution failure.

This local interface has no monitoring tool connection. It must not claim that
an alarm was checked unless a future scoped tool result confirms it.
