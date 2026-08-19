# EventBridge Archive Replay Operations

> **Synthetic-data notice:** Synthetic training and evaluation document. This is not official AWS documentation.

## Purpose and operating context

This guide explains how to replay a bounded historical event set after a consumer repair without duplicating successful effects. It is written for an operator who must preserve evidence, respect client and environment boundaries, and make a change only after its scope is understood. The recommended procedure assumes that raw inputs, derived artifacts, descriptors, and manifests have stable identities. It also assumes that retries can occur, so success is established from persisted state rather than from a single green invocation metric.

The focused control is: Select the narrowest time and event-pattern window, identify replayed events by original event ID, and require downstream idempotency before starting replay. The control should be expressed in configuration, policy, or state-transition logic that can be reviewed offline. Record the client, environment, namespace, domain, document or partition identity, and configuration version whenever those dimensions apply. That evidence makes a later incident traceable without logging document content, credentials, tokens, connection strings, or secret values.

## Control design

Treat the operation as a sequence of bounded transitions. First identify the authoritative input and its immutable checksum or version. Next identify the exact output scope and the condition that permits promotion. Then determine which failures are retryable and which require quarantine or human review. A retry must reuse the same operation identity; it must not invent another document, partition, chunk, execution side effect, or vector point.

For this subject, **replay scope** means the conjunction of archive, time window, and event pattern that defines exactly which historical events are delivered again. That definition matters because similar service terms can describe different evidence. An infrastructure status, event-delivery status, application descriptor, and business-data completeness signal are not interchangeable. Operators should cite the precise signal they used and should preserve a negative control that demonstrates adjacent client, environment, prefix, or state access remains denied where relevant.

## Standard procedure

1. **Freeze scope.** Write down the affected client, environment, data domain, time window, input identity, and current persisted state. Do not broaden the window merely to make a test convenient.
2. **Capture a baseline.** Record safe metadata, counts, checksums, state versions, alarms, and configuration fingerprints before changing or retrying anything. Historical DLQ messages and test artifacts are evidence, not cleanup targets.
3. **Apply the focused control.** Select the narrowest time and event-pattern window, identify replayed events by original event ID, and require downstream idempotency before starting replay. Change one variable at a time and keep production behavior opt-in when the control enables external or costly work.
4. **Exercise the reviewed path.** Use a deterministic fixture or the smallest authorized production event. Never substitute repeated uploads, broad replay, or wildcard permission for diagnosis.
5. **Reconcile outputs.** Record archive name, event pattern, UTC window, estimated event count, original IDs, target attempts, duplicate suppressions, failures, and final state counts. A document or partition is ready only when every required state and side effect is confirmed; absence of an error message is insufficient.

## Practical example

After a two-hour target outage, operators replay only document.ready events for client-blue/dev and compare original event IDs with descriptor completion records. The example deliberately keeps names synthetic and omits account numbers and credentials. The same method should be applied with the deployment's authorized configuration rather than copying identifiers from this document. When a service is at-least-once, duplicate delivery is expected input behavior; deterministic IDs and conditional state transitions are what keep the resulting data correct.

An operator should keep an evidence row containing the operation ID, input checksum, expected outputs, observed outputs, attempt count, and final state. If the operation touches multiple services, use one correlation ID but retain each service's native request or execution identifier separately. This prevents a successful handoff from being mistaken for successful downstream processing.

## Failure mode and remediation

**Observed failure:** Replay triggers processing for unaffected clients and produces duplicate downstream notifications.

**Likely cause:** The replay window and pattern were too broad, and the consumer used delivery time instead of original event ID for idempotency.

**Safe remediation:** Stop or contain the replay, reconcile affected IDs, correct the pattern and idempotency key, then run a dry inventory before one bounded replacement replay. Preserve the original inputs and failed state until reconciliation is complete. If a repair would create, replace, delete, replay, or invoke an external resource, stop at the review boundary and obtain the required approval. Do not retrieve secret payloads merely to prove that a secret reference exists; metadata and configuration validation should be used when the value is not needed.

A second common failure is declaring success from an aggregate count while one detailed item is missing. Compare the expected set with the actual set by deterministic identity, not only by total. A third failure is expanding IAM, a time window, or a namespace after AccessDenied or empty results. Trace the exact request and data path first; broadening scope can hide the cause and violate isolation.

## Monitoring, evidence, and warnings

The minimum operational evidence for this guide is: Record archive name, event pattern, UTC window, estimated event count, original IDs, target attempts, duplicate suppressions, failures, and final state counts. Useful overlapping terms for retrieval and incident correlation include EventBridge, archive, replay, event ID, idempotency, incident recovery. Expected chunk topics are bounded replay plan, original event identity, overbroad replay failure, reconciliation evidence. Metrics and logs must use safe identifiers and counts. They must not contain source record bodies, API keys, passwords, authorization headers, secret values, or customer contact information.

> **Warning:** Never mark a document, partition, or execution ready while required work is pending, failed, or unverifiable. Never assume that a retry is safe unless completed side effects are idempotent or explicitly detected. Never use a successful synthesis, API acceptance response, or empty DLQ as the sole proof of end-to-end correctness.

## Completion checklist

- The authoritative input and deterministic operation identity are recorded.
- Client, environment, namespace, and domain scope are explicit where applicable.
- The focused control is present without wildcard permissions or unrelated changes.
- Retryable and non-retryable failures have distinct state transitions.
- Detailed outputs reconcile to the aggregate descriptor or manifest.
- Monitoring covers stalled work, retries, failures, and final readiness.
- Logs and artifacts contain no credentials, tokens, secret payloads, or real customer data.
- A rollback or containment path is documented before external execution.
- The final evidence supports the claimed state and includes a negative isolation check.

This checklist is intentionally conservative. It favors a reproducible, narrowly scoped operation over a fast but ambiguous recovery. That tradeoff is appropriate for shared data platforms where one mistaken prefix, retry, schema update, or vector filter can affect multiple clients.
