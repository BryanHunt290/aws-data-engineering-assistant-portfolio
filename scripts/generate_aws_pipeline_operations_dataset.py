"""Deterministically generate the AWS pipeline operations dataset.

The generator is deliberately offline: it uses a reviewed, in-repository catalog
and never initializes application providers or network clients.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("data/aws_pipeline_operations")
DATASET_VERSION = "1.0.0"
SPLIT_SEED = "aws-pipeline-operations-v1-fixed-seed"
CREATED_AT = "2026-08-04T00:00:00Z"
LICENSE = "CC-BY-4.0"
SYNTHETIC_NOTICE = (
    "Synthetic training and evaluation document. This is not official AWS "
    "documentation."
)


@dataclass(frozen=True)
class DocumentSpec:
    slug: str
    title: str
    service: str
    domain: str
    document_type: str
    difficulty: str
    objective: str
    control: str
    example: str
    failure: str
    cause: str
    remediation: str
    verification: str
    term: str
    term_definition: str
    keywords: tuple[str, ...]
    chunk_topics: tuple[str, ...]

    @property
    def document_id(self) -> str:
        number = DOCUMENT_SPECS.index(self) + 1
        return f"apo-{number:03d}-{self.slug.replace('_', '-')}"

    @property
    def filename(self) -> str:
        return f"{self.slug}.md"


def _spec(
    slug: str,
    title: str,
    service: str,
    domain: str,
    document_type: str,
    difficulty: str,
    objective: str,
    control: str,
    example: str,
    failure: str,
    cause: str,
    remediation: str,
    verification: str,
    term: str,
    term_definition: str,
    keywords: str,
    chunk_topics: str,
) -> DocumentSpec:
    return DocumentSpec(
        slug=slug,
        title=title,
        service=service,
        domain=domain,
        document_type=document_type,
        difficulty=difficulty,
        objective=objective,
        control=control,
        example=example,
        failure=failure,
        cause=cause,
        remediation=remediation,
        verification=verification,
        term=term,
        term_definition=term_definition,
        keywords=tuple(value.strip() for value in keywords.split(",")),
        chunk_topics=tuple(value.strip() for value in chunk_topics.split("|")),
    )


DOCUMENT_SPECS = (
    _spec(
        "s3_prefix_isolation",
        "S3 Prefix Isolation for Shared Pipeline Buckets",
        "Amazon S3",
        "security",
        "security_policy",
        "intermediate",
        "separate client and environment data inside a shared pipeline bucket without granting visibility into neighboring prefixes",
        "Bind object actions to the exact client/environment namespace and bind ListBucket to the bucket ARN with an s3:prefix condition containing only that namespace.",
        "A dev loader for client-blue writes under knowledge/client-blue/dev/raw/ and may list that prefix, but it cannot read knowledge/client-green/ or the production namespace.",
        "A loader can fetch known keys but receives AccessDenied while checking whether a generated metadata object exists.",
        "GetObject was scoped correctly, but ListBucket was omitted or its prefix condition did not include the metadata path used by the existence check.",
        "Add only the missing metadata prefix to the ListBucket condition, retain object actions on object ARNs, and re-run both allowed and denied policy tests.",
        "Inspect the synthesized policy, test one permitted prefix and one adjacent denied prefix, and record the client, environment, namespace, and object key in the validation evidence.",
        "prefix isolation",
        "a policy boundary in which an identity can operate only inside its assigned key hierarchy even when the physical bucket is shared",
        "S3,IAM,prefix,ListBucket,GetObject,client isolation,namespace",
        "bucket and object ARN distinction|prefix-conditioned listing|negative authorization tests|client namespace evidence",
    ),
    _spec(
        "s3_lifecycle_cost_controls",
        "S3 Lifecycle Cost Controls for Pipeline Artifacts",
        "Amazon S3",
        "cost_optimization",
        "architecture_guide",
        "intermediate",
        "reduce storage cost for raw, processed, and diagnostic artifacts without expiring evidence before its retention obligation ends",
        "Classify prefixes by recovery value, set explicit retention owners, and apply lifecycle transitions only after downstream replay and audit windows have elapsed.",
        "Raw inputs remain immediately available for thirty days, transition to a colder class after validation, and expire only after the replay window; manifests and incident evidence retain longer schedules.",
        "A replay requested during an incident cannot find the original raw object even though derived chunks still exist.",
        "A broad lifecycle rule expired the entire knowledge/ hierarchy instead of targeting disposable intermediate artifacts.",
        "Suspend the affected rule, restore recoverable versions where available, reconstruct only from verified sources, and replace the rule with prefix- and tag-scoped retention.",
        "Use lifecycle-rule previews, inventory age distributions, noncurrent-version counts, and a restore drill before approving expiration behavior.",
        "retention horizon",
        "the longest operational, legal, or replay interval that an artifact must survive before transition or expiration is safe",
        "S3,lifecycle,retention,cost,replay,storage class,artifacts",
        "artifact classification|transition timing|expiration failure|restore and cost evidence",
    ),
    _spec(
        "s3_event_delivery_diagnostics",
        "Diagnosing S3 Object-Created Event Delivery",
        "Amazon S3",
        "incident_response",
        "troubleshooting_runbook",
        "advanced",
        "determine why an expected object-created event did not produce one successful ingestion without creating duplicate test traffic",
        "Trace a single unique object key through notification filters, Lambda permission, invocation metrics, logs, retry state, and the dead-letter queue in that order.",
        "An operator uploads one object under knowledge/raw/smoke-tests/ with a unique correlation suffix and compares the upload time with the function invocation window.",
        "The object exists in S3, but there is no matching request ID or ingestion log entry.",
        "The key did not match the configured prefix or suffix filter, or the bucket lacked permission to invoke the target function.",
        "Do not upload another object first; inspect notification metadata and invoke permission, correct only the mismatched filter or permission, then use one new unique key for validation.",
        "Confirm exactly one asynchronous invocation, no retry metrics, no new DLQ message, and the expected processed, chunk, metadata, and descriptor objects.",
        "event delivery",
        "the at-least-once handoff from an S3 notification configuration to an authorized target, distinct from successful application processing",
        "S3,event notification,Lambda,DLQ,retry,correlation,smoke test",
        "notification filters|invoke permission|single-event trace|retry and DLQ evidence",
    ),
    _spec(
        "glue_job_bookmark_recovery",
        "Glue Job Bookmark Recovery Without Duplicate Loads",
        "AWS Glue",
        "reliability",
        "standard_operating_procedure",
        "advanced",
        "recover a stalled incremental Glue job while preserving a clear boundary between previously committed and pending source data",
        "Capture the current bookmark and sink commit evidence before any reset, then replay a bounded source interval into an idempotent staging path.",
        "A daily job last committed partition date=2026-07-30; operators stage only the 2026-07-31 partition, reconcile keys, and promote it after counts match.",
        "Resetting the bookmark causes historical partitions to be emitted again and duplicate business keys appear downstream.",
        "The reset broadened the read horizon while the sink used append-only writes without a deterministic merge key.",
        "Pause promotion, quarantine replay output, restore the captured bookmark or bounded predicate, deduplicate by the documented business key, and resume from the first uncommitted partition.",
        "Reconcile bookmark position, input partitions, staged row counts, distinct business keys, sink commit IDs, and the next scheduled run before closing recovery.",
        "bookmark boundary",
        "the persisted source progress point that separates data a Glue job considers processed from data eligible for the next incremental run",
        "Glue,job bookmark,incremental load,replay,idempotency,partition",
        "bookmark evidence|bounded replay|duplicate prevention|recovery reconciliation",
    ),
    _spec(
        "glue_schema_evolution_contract",
        "Glue Schema Evolution Contract for Curated Tables",
        "AWS Glue",
        "schema_evolution",
        "architecture_guide",
        "advanced",
        "allow compatible source changes while preventing silent type drift from corrupting curated analytical tables",
        "Compare each discovered schema with a versioned contract and route incompatible additions, removals, or type changes to quarantine before catalog publication.",
        "Adding a nullable campaign_code column is accepted after consumer review, while changing amount from decimal to string fails validation and keeps the prior catalog schema active.",
        "Athena queries begin returning nulls for a formerly populated numeric column after a crawler run.",
        "The crawler updated the catalog from mixed source files without enforcing the curated table's type-compatibility contract.",
        "Freeze catalog updates, identify the first incompatible partition, quarantine it, restore the approved schema version, and require an explicit migration for the producer change.",
        "Record schema fingerprints, compatibility decisions, quarantined object counts, catalog version, and consumer approval in the deployment evidence.",
        "compatible evolution",
        "a schema change that preserves existing consumer interpretations under the repository's declared nullability, type, and field-removal rules",
        "Glue,Data Catalog,schema evolution,compatibility,quarantine,Athena",
        "versioned schema contract|compatible additions|type drift incident|catalog recovery evidence",
    ),
    _spec(
        "glue_worker_capacity_tuning",
        "Glue Worker Capacity Tuning With Evidence",
        "AWS Glue",
        "cost_optimization",
        "service_comparison",
        "intermediate",
        "choose Glue worker capacity from measured stage behavior rather than permanently overprovisioning every job",
        "Baseline input size, shuffle, skew, spill, duration, and cost per successful partition, then change one worker dimension at a time.",
        "A join-heavy job with one hot customer key is first salted to reduce skew; only then are worker count trials compared against the same frozen input partitions.",
        "Doubling workers raises cost but does not reduce wall-clock duration or executor loss.",
        "A skewed transformation serializes work on a few partitions, so additional workers remain idle and cannot remove the bottleneck.",
        "Profile stage distribution, correct skew or file sizing, rerun the baseline, and select the smallest capacity that meets the duration and failure-rate objective.",
        "Compare median and tail duration, DPU-hours, spill volume, executor failures, input bytes, and output file distribution across repeated equivalent runs.",
        "capacity efficiency",
        "successful processed input per unit of worker time and cost, evaluated together with reliability rather than as maximum cluster size",
        "Glue,workers,DPU,cost,skew,shuffle,performance",
        "baseline metrics|skew diagnosis|controlled capacity trial|cost and duration decision",
    ),
    _spec(
        "athena_partition_projection",
        "Athena Partition Projection Operations Guide",
        "Amazon Athena",
        "query_operations",
        "architecture_guide",
        "advanced",
        "use deterministic partition rules for predictable query planning while preventing projected paths that do not exist or cross client boundaries",
        "Define projection ranges and storage templates from the physical S3 layout, include client and environment constraints, and test boundary dates before enabling the table property.",
        "A table projects day values for a rolling window and maps client-blue/dev/day=${day} to exactly one isolated S3 prefix; a future boundary query returns zero rows without scanning another client.",
        "Queries scan many empty prefixes and become slower after projection is enabled.",
        "The projected date range is unbounded or the storage template does not match the actual partition granularity.",
        "Narrow the range, correct the template, run EXPLAIN and boundary queries, and compare scanned bytes with the previous catalog-partition plan before rollout.",
        "Verify template expansion, minimum and maximum partition values, client isolation, scanned bytes, query plan, and representative result counts.",
        "projected partition",
        "a partition value Athena derives from table properties at query time instead of discovering it as a catalog partition record",
        "Athena,partition projection,S3,query plan,scanned bytes,isolation",
        "projection rules|storage template|empty-prefix failure|boundary query evidence",
    ),
    _spec(
        "athena_query_cost_guardrails",
        "Athena Query Cost Guardrails and Workgroup Controls",
        "Amazon Athena",
        "cost_optimization",
        "security_policy",
        "intermediate",
        "bound ad hoc query cost without preventing legitimate incident analysis or exposing another environment's results",
        "Use isolated workgroups with enforced result locations, per-query scan cutoffs, tagged ownership, and alarms on aggregate bytes scanned.",
        "The internal-dev workgroup rejects a query after its scan threshold rather than allowing an accidental unpartitioned full-table read; the operator rewrites it with date and client predicates.",
        "A dashboard refresh repeatedly reaches the scan cutoff even though its result set is small.",
        "The query filters on a derived expression that prevents partition pruning and reads wide text files instead of selected columnar fields.",
        "Inspect EXPLAIN output, add direct partition predicates, select required columns, compact or convert source files when justified, and remeasure scanned bytes.",
        "Review workgroup configuration, result-prefix isolation, query history, bytes scanned, cutoff failures, and cost allocation tags for the same period.",
        "scan cutoff",
        "a workgroup-enforced maximum amount of data one query may read before Athena terminates it as a cost-control action",
        "Athena,workgroup,scan cutoff,cost,partition pruning,result location",
        "workgroup isolation|scan limits|partition-pruning failure|cost evidence",
    ),
    _spec(
        "athena_result_location_security",
        "Securing Athena Query Result Locations",
        "Amazon Athena",
        "security",
        "security_policy",
        "intermediate",
        "keep query results, metadata, and temporary artifacts within the requesting client and environment boundary",
        "Enforce the result location through the workgroup, deny caller overrides, encrypt results, and scope both S3 object and listing permissions to the result namespace.",
        "Queries for client-green/dev write only to analytics-results/client-green/dev/ and the execution role cannot list or fetch client-green/prod/ results.",
        "A user can submit a query that writes output to a personal or cross-client bucket prefix.",
        "The workgroup does not enforce its configured settings, allowing the API request to override the output location.",
        "Enable workgroup enforcement, remove broader result-bucket grants, expire misplaced temporary results according to policy, and test a denied override request.",
        "Inspect workgroup enforcement, output ARN, encryption setting, bucket policy, object ownership, lifecycle retention, and negative cross-prefix access tests.",
        "enforced result configuration",
        "a workgroup setting that makes centrally reviewed output and encryption controls authoritative over caller-supplied query options",
        "Athena,S3,workgroup,query results,encryption,least privilege",
        "result namespace|workgroup enforcement|override failure|negative access evidence",
    ),
    _spec(
        "lambda_s3_event_idempotency",
        "Idempotent Lambda Processing for S3 Events",
        "AWS Lambda",
        "reliability",
        "architecture_guide",
        "advanced",
        "make repeated or out-of-order S3 notifications safe while retaining enough state to resume incomplete document processing",
        "Derive a stable operation key from bucket, decoded object key, object version or immutable checksum, and processing schema version; persist completion with conditional writes.",
        "Two notifications for the same immutable raw object resolve to one document identifier; the second reads a ready descriptor and performs no extraction, embedding, or vector write.",
        "A retry creates duplicate chunks and overwrites a ready manifest entry with pending state.",
        "The handler generated a random identifier per event and treated the aggregate descriptor as an unconditional last-writer-wins object.",
        "Stop retries if they amplify damage, restore the last checksum-valid descriptor, adopt deterministic IDs and conditional transitions, and redrive only missing chunk states.",
        "Send the same fixture event twice in an offline test and confirm stable document and chunk IDs, unchanged ready states, one vector per chunk, and no duplicated manifest entry.",
        "idempotency key",
        "a deterministic identity for one logical input operation that lets repeated deliveries converge on the same persisted state",
        "Lambda,S3,idempotency,conditional write,retry,descriptor,checksum",
        "stable event identity|conditional state transition|duplicate incident|replay verification",
    ),
    _spec(
        "lambda_concurrency_backpressure",
        "Lambda Concurrency and Pipeline Backpressure",
        "AWS Lambda",
        "reliability",
        "operations_guide",
        "advanced",
        "control ingestion pressure without exhausting account concurrency or overwhelming downstream storage and model quotas",
        "Model arrival rate, duration, retry amplification, and downstream capacity before setting any reserved concurrency; omit it when a low account quota cannot preserve required unreserved capacity.",
        "An internal-dev account with a small concurrency quota leaves reserved concurrency unset and uses alarms plus bounded asynchronous retries rather than reserving a positive value it cannot support.",
        "A stack update rolls back because the function requests reserved concurrency while the account must retain the entire small quota as unreserved.",
        "Environment-specific quota constraints were ignored and a production-style fixed concurrency value was synthesized for development.",
        "Remove the internal-dev reservation entirely, keep configurability for approved environments, synthesize both modes, and review the resulting Lambda property before deployment.",
        "Confirm ReservedConcurrentExecutions is absent, runtime safeguards remain unchanged, account concurrency metrics are stable, and DLQ age is not increasing.",
        "backpressure",
        "a deliberate limit or delay that keeps incoming work within the safe processing rate of Lambda and its downstream dependencies",
        "Lambda,concurrency,backpressure,quota,DLQ,retry,CDK",
        "capacity model|environment override|quota deployment failure|template and metric evidence",
    ),
    _spec(
        "lambda_dependency_packaging",
        "Packaging Python Dependencies for Lambda",
        "AWS Lambda",
        "deployment_safety",
        "deployment_checklist",
        "advanced",
        "produce a reproducible Python dependency artifact compatible with the Lambda runtime and architecture",
        "Install fully pinned dependencies for the target Linux platform into a python/ layer directory, inspect native libraries, hash the ZIP, and publish only the reviewed bytes.",
        "A Python 3.12 x86_64 layer is built from a lock file, contains python/qdrant_client and transitive distributions, has no Windows binaries, and is identified by a recorded SHA-256.",
        "The function fails during initialization with an ImportModuleError for a transitive package that was available on the developer workstation.",
        "The artifact included only the top-level package or was built against a different platform, so dependency closure was incomplete in Lambda.",
        "Do not patch the live function; rebuild in an ignored directory from the reviewed lock, validate imports without network clients, compare the archive manifest, and publish a new reviewed version only with approval.",
        "Check runtime, architecture, python/ root structure, distribution versions, native library formats, compressed and uncompressed size, import closure, and archive checksum.",
        "dependency closure",
        "the complete set of direct and transitive packages required for imports to succeed in the target Lambda runtime",
        "Lambda,Python 3.12,layer,dependency lock,archive,checksum,ABI",
        "pinned build|layer structure|missing dependency failure|archive inspection evidence",
    ),
    _spec(
        "step_functions_retry_policy",
        "Step Functions Retry Policy Design",
        "AWS Step Functions",
        "reliability",
        "architecture_guide",
        "advanced",
        "retry transient pipeline failures without repeating non-idempotent work or hiding permanent validation errors",
        "Classify errors by retryability, use bounded exponential backoff with jitter where supported, catch permanent failures separately, and require idempotent task operations.",
        "A throttled service task retries three times with increasing delay, while an InvalidSchema error bypasses retries and writes a failed descriptor with remediation context.",
        "A malformed document consumes the maximum retries on every execution and floods the downstream queue.",
        "The retry rule matched States.ALL before a specific non-retryable validation error could be caught.",
        "Reorder error handling, exclude validation classes from retry, preserve failure context, and redrive only after the document or contract has been corrected.",
        "Use execution history to verify attempt count, interval progression, matched error class, catch target, idempotency key, and final descriptor state.",
        "retryable error",
        "a failure expected to succeed without changing the input or contract, such as bounded throttling or a temporary dependency outage",
        "Step Functions,retry,backoff,idempotency,validation,execution history",
        "error classification|bounded backoff|retry storm failure|execution-history evidence",
    ),
    _spec(
        "step_functions_redrive_safety",
        "Safe Step Functions Execution Redrive",
        "AWS Step Functions",
        "incident_response",
        "standard_operating_procedure",
        "advanced",
        "resume a failed state-machine execution without repeating completed external side effects",
        "Inventory completed task outputs and idempotency records before redrive, retain the original input, and resume only states whose persisted completion is absent or invalid.",
        "A document has four chunks, three ready states, and one retryable failure; redrive reuses the three ready embeddings and writes only the missing vector.",
        "Redrive repeats a completed publish task and produces two downstream records for the same chunk.",
        "The task did not consult its deterministic operation key or persisted completion marker before performing the side effect.",
        "Pause further redrives, reconcile duplicate outputs, add the completion guard, verify descriptor checksums, and redrive a single controlled execution.",
        "Compare original and redriven execution IDs, resumed states, task attempt counts, skipped completed chunks, final descriptor totals, and manifest readiness.",
        "redrive boundary",
        "the set of incomplete or failed states eligible to run again while previously successful side effects remain authoritative",
        "Step Functions,redrive,resume,idempotency,descriptor,vector state",
        "pre-redrive inventory|completed-state reuse|duplicate side effect|redrive reconciliation",
    ),
    _spec(
        "step_functions_distributed_map_controls",
        "Distributed Map Controls for Large Data Sets",
        "AWS Step Functions",
        "scalability",
        "architecture_guide",
        "advanced",
        "parallelize large object sets without losing per-item identity, cost bounds, or failure visibility",
        "Partition input into immutable item manifests, set concurrency from downstream capacity, store item-level outcomes, and define tolerated failure thresholds explicitly.",
        "A map processes a frozen manifest of partition files with each item carrying client, environment, object checksum, and operation ID; failed items are exported for bounded redrive.",
        "The map completes with an acceptable aggregate status even though critical client partitions were among tolerated failures.",
        "A global failure percentage treated all items equally and did not enforce zero tolerance for designated critical partitions.",
        "Separate critical items or add a post-map gate, preserve failed item manifests, correct the cause, and redrive only the identified failures.",
        "Review manifest checksum, item count, concurrency, downstream throttles, failure threshold, critical-item gate, per-item results, and execution cost.",
        "tolerated failure threshold",
        "the explicit number or percentage of failed map items permitted before the distributed operation is considered failed",
        "Step Functions,Distributed Map,concurrency,manifest,redrive,cost",
        "immutable item manifest|parallelism control|critical failure gap|item-level evidence",
    ),
    _spec(
        "eventbridge_schema_routing",
        "EventBridge Schema-Aware Pipeline Routing",
        "Amazon EventBridge",
        "event_driven_architecture",
        "architecture_guide",
        "intermediate",
        "route versioned pipeline events to the correct client and environment handlers without brittle string matching",
        "Match a stable envelope containing event_type, schema_version, client, environment, and domain; validate payload contracts in the target before side effects.",
        "A pipeline.document.ready version 2 event for client-blue/dev routes to the dev indexing target, while version 3 is quarantined until a compatible rule and consumer exist.",
        "A newly versioned event silently stops reaching its consumer even though it is present on the bus.",
        "The rule pattern constrained schema_version to the prior value and no unsupported-version alarm or archive review was configured.",
        "Confirm the producer version, add an explicitly reviewed compatible target or quarantine route, replay one archived fixture, and monitor matched and failed invocations.",
        "Inspect rule patterns, target ARNs, input transformation, schema compatibility, client/environment fields, match metrics, target errors, and DLQ state.",
        "event envelope",
        "the stable routing metadata around a versioned domain payload, used to isolate and validate events before processing",
        "EventBridge,schema version,routing,event envelope,client isolation,DLQ",
        "routing envelope|version compatibility|unmatched event incident|rule and target evidence",
    ),
    _spec(
        "eventbridge_archive_replay",
        "EventBridge Archive Replay Operations",
        "Amazon EventBridge",
        "incident_response",
        "standard_operating_procedure",
        "advanced",
        "replay a bounded historical event set after a consumer repair without duplicating successful effects",
        "Select the narrowest time and event-pattern window, identify replayed events by original event ID, and require downstream idempotency before starting replay.",
        "After a two-hour target outage, operators replay only document.ready events for client-blue/dev and compare original event IDs with descriptor completion records.",
        "Replay triggers processing for unaffected clients and produces duplicate downstream notifications.",
        "The replay window and pattern were too broad, and the consumer used delivery time instead of original event ID for idempotency.",
        "Stop or contain the replay, reconcile affected IDs, correct the pattern and idempotency key, then run a dry inventory before one bounded replacement replay.",
        "Record archive name, event pattern, UTC window, estimated event count, original IDs, target attempts, duplicate suppressions, failures, and final state counts.",
        "replay scope",
        "the conjunction of archive, time window, and event pattern that defines exactly which historical events are delivered again",
        "EventBridge,archive,replay,event ID,idempotency,incident recovery",
        "bounded replay plan|original event identity|overbroad replay failure|reconciliation evidence",
    ),
    _spec(
        "cloudwatch_pipeline_observability",
        "CloudWatch Observability for Document Pipelines",
        "Amazon CloudWatch",
        "observability",
        "operations_guide",
        "intermediate",
        "show whether documents progress from received through ready while keeping logs free of content and credentials",
        "Emit structured stage events with correlation IDs and safe dimensions, derive latency and state-count metrics, and alarm on stalled work rather than only function errors.",
        "Logs record document_id, client, environment, stage, elapsed_seconds, success, and error_type; they never record document text, endpoint credentials, or secret values.",
        "Lambda error rate is zero, yet pending descriptors accumulate for hours and users see stale retrieval results.",
        "The dashboard observes invocation health but has no age or state-transition metric for application-level backlog.",
        "Add pending-age and transition-rate metrics from safe state records, alarm on the oldest pending item, and investigate the first stalled stage using its correlation ID.",
        "Verify log schema, redaction tests, received-to-ready latency, pending age, failed counts, retry counts, DLQ age, alarm actions, and dashboard client filters.",
        "stage latency",
        "elapsed time between two persisted pipeline state transitions, not merely the duration of one Lambda invocation",
        "CloudWatch,structured logs,metrics,alarm,pending age,correlation ID",
        "safe telemetry schema|state metrics|silent backlog incident|dashboard and alarm evidence",
    ),
    _spec(
        "cloudwatch_alarm_runbook",
        "Pipeline Alarm Triage Runbook",
        "Amazon CloudWatch",
        "incident_response",
        "troubleshooting_runbook",
        "intermediate",
        "triage a pipeline alarm consistently and gather evidence before retrying, suppressing, or changing infrastructure",
        "Start from the alarm state transition and affected dimension, correlate metrics with logs and persisted descriptors, then classify impact and retryability.",
        "A failed-chunk alarm identifies client-blue/dev; the operator finds one ValidationError, confirms no retry amplification, quarantines that document, and leaves healthy clients running.",
        "An operator treats an INSUFFICIENT_DATA transition as recovery and closes the incident while the metric publisher remains broken.",
        "Missing datapoints were interpreted as zero failures even though the alarm's missing-data policy did not support that conclusion.",
        "Restore or validate metric publication, inspect the alarm's missing-data behavior, keep the incident open until fresh points arrive, and document the observed state timeline.",
        "Capture alarm ARN without account-specific examples, state reason, UTC timestamps, dimensions, linked log request IDs, descriptor status, DLQ counts, and remediation owner.",
        "missing data",
        "an absence of metric samples whose operational meaning depends on the alarm configuration and must not automatically be read as healthy",
        "CloudWatch,alarm,triage,missing data,logs,descriptor,DLQ",
        "alarm context|correlated investigation|missing-data trap|incident evidence checklist",
    ),
    _spec(
        "iam_least_privilege_data_pipeline",
        "Least-Privilege IAM for Data Pipelines",
        "AWS IAM",
        "security",
        "security_policy",
        "advanced",
        "grant each pipeline component only the actions and resources required for its reviewed data transition",
        "Derive permissions from explicit read, write, list, invoke, and metadata operations; separate bucket-level from object-level ARNs and constrain model and secret access to named resources.",
        "An ingestion role reads one raw object, writes processed and chunk artifacts, reads only its approved secret ARN when indexing is enabled, and invokes one approved embedding model ARN.",
        "A deployment diff adds Resource star to make a failing task pass even though the task touches one known bucket prefix.",
        "Permission troubleshooting was performed by broadening first instead of tracing the exact API, resource type, condition, and caller.",
        "Reject the broad change, capture the denied API from metadata-safe evidence, add the narrow resource and condition, and test both the intended operation and a neighboring denied operation.",
        "Review every action-resource pair, conditions, service-specific ARN shape, cross-account principals, wildcard absence, negative tests, and the authenticated diff.",
        "least privilege",
        "the smallest reviewed set of actions, resources, and conditions that permits an intended workload while denying adjacent operations",
        "IAM,least privilege,policy,S3,Bedrock,Secrets Manager,negative test",
        "permission inventory|resource scoping|wildcard incident|policy and negative-test evidence",
    ),
    _spec(
        "iam_cross_client_isolation",
        "IAM Controls for Cross-Client Isolation",
        "AWS IAM",
        "security",
        "security_policy",
        "advanced",
        "prevent identities assigned to one client from enumerating, reading, writing, or invoking resources for another client",
        "Carry immutable client and environment attributes into role selection, resource names, prefixes, and policy conditions, with explicit deny guardrails for mismatches.",
        "A client-blue/dev session can write its staging prefix and query its workgroup, while attempts against client-green or prod fail even when object names are guessed.",
        "A support role intended for one development client can list object keys across the shared bucket.",
        "Its object permissions were scoped, but its bucket-level ListBucket action lacked a client prefix condition.",
        "Remove unrestricted listing, add the exact client/environment prefix condition, review session-tag trust boundaries, and run adjacent-client and adjacent-environment denial tests.",
        "Validate principal tags, trust policy, role path, bucket list conditions, object ARNs, workgroup scope, secret ARN, and negative matrix results.",
        "client context",
        "the authenticated, immutable client and environment attributes used to select and constrain every downstream data-plane operation",
        "IAM,client isolation,principal tag,explicit deny,S3 prefix,environment",
        "trusted client context|policy propagation|cross-client listing failure|isolation test matrix",
    ),
    _spec(
        "data_quality_quarantine_workflow",
        "Data Quality Quarantine Workflow",
        "AWS Glue",
        "data_quality",
        "standard_operating_procedure",
        "intermediate",
        "separate invalid records from accepted data while retaining reproducible evidence and preventing accidental promotion",
        "Evaluate versioned rules before publication, write failures to an isolated quarantine prefix with reason codes, and promote only a checksum-matched accepted set.",
        "Rows with negative quantity receive rule_id quantity_nonnegative and stay under quarantine/client-blue/dev/run-42/, while accepted rows and a signed-off summary advance together.",
        "Corrected data is reprocessed, but stale quarantine records are accidentally included in the curated output.",
        "The promotion job selected a broad run prefix instead of the immutable accepted-object manifest for the corrected attempt.",
        "Stop publication, identify affected partitions, restore the prior curated snapshot, generate a new accepted manifest, and promote only checksums listed in that manifest.",
        "Reconcile input, accepted, rejected, and missing counts; rule versions; reason distributions; manifest checksums; approval; and curated partition totals.",
        "quarantine",
        "an isolated, non-promotable holding area for records that failed a named data-quality or schema rule and retain diagnostic context",
        "data quality,quarantine,rule version,manifest,checksum,promotion",
        "rule evaluation|quarantine evidence|stale-record incident|promotion reconciliation",
    ),
    _spec(
        "data_quality_freshness_monitoring",
        "Freshness Monitoring for Partitioned Data",
        "Amazon CloudWatch",
        "data_quality",
        "operations_guide",
        "intermediate",
        "detect when expected partitions or events arrive late without confusing an intentionally quiet source with an outage",
        "Define freshness against event time and an explicit delivery schedule per dataset, then alarm on watermark lag with maintenance and holiday exceptions.",
        "A dataset due hourly allows fifteen minutes of lateness; its watermark comes from the maximum validated event timestamp, not the S3 object's upload time.",
        "The freshness alarm remains green although yesterday's business data is missing.",
        "Recent retry files advanced the object-arrival metric while the validated event-time watermark did not advance.",
        "Change the signal to the accepted-data watermark, backfill the missing partition through the idempotent path, and record the cause of the delivery delay.",
        "Compare schedule, expected partition, event-time watermark, object arrival, validation completion, lag threshold, exception calendar, and downstream query visibility.",
        "watermark lag",
        "the difference between the current evaluation time and the latest validated event time that is safe for downstream consumption",
        "data quality,freshness,watermark,partition,CloudWatch,late data",
        "freshness contract|event-time watermark|false-green incident|lag and partition evidence",
    ),
    _spec(
        "schema_registry_compatibility",
        "Schema Registry Compatibility Operations",
        "AWS Glue Schema Registry",
        "schema_evolution",
        "operations_guide",
        "advanced",
        "evolve event schemas under an explicit compatibility mode while keeping producers and consumers independently deployable",
        "Register schemas with stable names, test changes against the selected backward or forward contract, and deploy consumers before producers when the compatibility direction requires it.",
        "A backward-compatible producer adds an optional field with a default; old data remains readable and consumers that ignore the field continue to work.",
        "A producer deployment is rejected because it removes a required field used by current consumers.",
        "The proposed version violates the registry's compatibility mode and no staged consumer migration has removed the dependency.",
        "Keep the existing producer version, inventory consumers, deploy a tolerant consumer contract, register a migration version, and only then remove the field under a reviewed policy.",
        "Record registry and schema names, compatibility mode, prior and proposed fingerprints, compatibility result, consumer inventory, deployment order, and rollback version.",
        "backward compatibility",
        "the ability of the new schema and reader plan to process data produced under prior approved schema versions according to declared field rules",
        "Glue Schema Registry,schema,compatibility,producer,consumer,version",
        "compatibility mode|optional field example|breaking removal|migration evidence",
    ),
    _spec(
        "pipeline_idempotency_keys",
        "Designing Pipeline Idempotency Keys",
        "AWS Lambda",
        "reliability",
        "architecture_guide",
        "advanced",
        "define stable identities that make retries and redrives converge across ingestion, transformation, embedding, and vector storage",
        "Compose keys from immutable business scope and input version, include processing schema where output meaning changes, and persist state with conditional transitions.",
        "A chunk vector ID derives from client, environment, namespace, domain, document checksum, chunk index, and chunk checksum, so a retry upserts the same point.",
        "Two different document revisions are treated as the same operation and the newer content never reaches ready state.",
        "The key used only filename and client, omitting the immutable object version or checksum that distinguishes revisions.",
        "Preserve both source revisions, update the key contract to include content identity, migrate state explicitly, and test duplicate delivery separately from legitimate revision processing.",
        "Verify canonical serialization, stable hash, collision tests, revision distinction, retry convergence, conditional writes, vector IDs, descriptor state, and manifest totals.",
        "operation identity",
        "the canonical immutable inputs that distinguish one logical unit of work from a retry of that same work",
        "idempotency,key,checksum,document revision,chunk,vector ID,conditional write",
        "canonical key inputs|deterministic vector identity|revision collision|retry convergence tests",
    ),
    _spec(
        "incident_missing_partitions",
        "Incident Response for Missing Partitions",
        "Amazon Athena",
        "incident_response",
        "incident_report",
        "intermediate",
        "restore query completeness when an expected partition is absent while preserving the distinction between late, rejected, and undiscovered data",
        "Trace the partition from source schedule through S3 arrival, validation, catalog or projection visibility, and consumer query predicates before initiating backfill.",
        "The 2026-07-31 partition exists in raw storage but all records are quarantined for a schema error, so adding a catalog entry would expose no valid data.",
        "Athena returns no rows for one date even though operators manually added that date to the catalog.",
        "The physical accepted-data prefix is empty because validation failed; catalog presence was mistaken for data completeness.",
        "Remove misleading manual state if needed, correct or approve the source schema, replay through validation, publish the accepted manifest, and verify the partition query.",
        "Record expected schedule, raw object count, quarantine reason, accepted manifest, catalog or projection mapping, query predicate, row count, and freshness watermark.",
        "partition completeness",
        "the condition in which the expected validated records for a partition are physically present, discoverable, and queryable, not merely cataloged",
        "Athena,partition,incident,backfill,quarantine,freshness,Glue Catalog",
        "partition trace|quarantined example|catalog-only false fix|backfill verification",
    ),
    _spec(
        "incident_duplicate_records",
        "Incident Response for Duplicate Records",
        "AWS Glue",
        "incident_response",
        "troubleshooting_runbook",
        "advanced",
        "contain and repair duplicate pipeline outputs without deleting evidence or reprocessing unaffected partitions",
        "Define the business and operation keys, freeze promotion for affected scope, identify the first duplicate-producing run, and rebuild from an immutable accepted manifest.",
        "A retried append job wrote order_id and event_version pairs twice in one partition; operators quarantine that partition and regenerate it with a deterministic merge.",
        "Row counts double after a timeout even though the orchestrator reports the first task as failed.",
        "The sink commit succeeded before the timeout, but the retry could not detect completion and appended the same operation again.",
        "Stop the retry source, preserve both attempts, deduplicate using the documented key and source order, replace only affected partitions, and add a conditional completion record.",
        "Reconcile raw and curated unique keys, duplicate groups, run IDs, commit timestamps, partition checksums, downstream extracts, and the next retry simulation.",
        "duplicate scope",
        "the exact client, dataset, partition, business key, and operation interval affected by repeated writes",
        "duplicates,incident,idempotency,business key,partition,retry,Glue",
        "containment boundary|append retry example|ambiguous commit failure|repair reconciliation",
    ),
    _spec(
        "cost_optimization_storage_compute",
        "Balancing Storage and Compute Cost in Data Lakes",
        "Amazon S3 and Amazon Athena",
        "cost_optimization",
        "service_comparison",
        "intermediate",
        "reduce total pipeline cost by treating file layout, retention, transformation, and query scans as one system",
        "Measure cost per accepted gigabyte and per useful query, then optimize small files, compression, columnar layout, partitioning, and lifecycle in that order of demonstrated benefit.",
        "A daily compaction step combines thousands of tiny JSON objects into bounded Parquet files, raising one processing cost while reducing repeated Athena scans and request overhead.",
        "Storage cost falls after aggressive compression, but query latency and Glue failures rise sharply.",
        "Files became too large for practical parallelism and the compression codec consumed disproportionate worker CPU for the access pattern.",
        "Restore a balanced file target, benchmark representative queries and jobs, retain source data through the rollback window, and choose the lowest total cost that meets reliability objectives.",
        "Track object count and size distribution, storage-class bytes, request counts, DPU-hours, scanned bytes, query latency, failed runs, and cost per accepted partition.",
        "total pipeline cost",
        "the combined storage, request, transformation, orchestration, and query expense required to produce and use validated data",
        "S3,Athena,Glue,cost,Parquet,small files,compression,partitioning",
        "cost model|compaction example|overcompression failure|cross-service cost evidence",
    ),
    _spec(
        "multi_client_namespace_isolation",
        "Multi-Client Namespace Isolation Contract",
        "AWS Data Pipeline",
        "multi_client_isolation",
        "architecture_guide",
        "advanced",
        "carry client, environment, namespace, and domain isolation through storage keys, descriptors, embeddings, vector collections, and retrieval filters",
        "Validate scope at configuration boundaries and include it in deterministic identifiers, physical prefixes, vector payloads, collection selection, logs, and authorization tests.",
        "The same document filename uploaded by client-blue/dev and client-green/dev produces different document IDs, prefixes, vector IDs, payload filters, and retrieval results.",
        "A retrieval request returns a semantically relevant chunk owned by another client.",
        "The vector search filter constrained domain but omitted client and environment, while the shared collection contained multiple tenants.",
        "Disable the unsafe query path, audit affected requests, require all isolation fields in filters and payload validation, reindex missing scope safely, and run cross-client negative tests.",
        "Inspect canonical scope, collection name, payload fields, filter construction, storage prefixes, IAM boundary, logs, deterministic IDs, and denial/retrieval test matrix.",
        "namespace",
        "a logical subdivision within a client and environment that must remain part of data identity, indexing payload, and retrieval authorization",
        "client,environment,namespace,domain,isolation,Qdrant,vector filter",
        "scope contract|deterministic scoped identity|cross-client retrieval incident|isolation verification matrix",
    ),
    _spec(
        "production_deployment_safety",
        "Production Deployment Safety for Data Pipelines",
        "AWS CloudFormation",
        "deployment_safety",
        "deployment_checklist",
        "advanced",
        "move a reviewed pipeline change from deterministic synthesis through authenticated comparison and explicit deployment approval",
        "Pin the configuration and assets, validate offline, verify identity, review every change and replacement indicator, execute only the approved stack or change set, and stop at the authorized milestone.",
        "A change set contains one Lambda code update and one bounded IAM modification, with zero additions, deletions, or replacements; its template hash and asset hash match recorded evidence.",
        "CloudFormation reports an unexpected metadata resource modification despite an apparently equivalent application template.",
        "The deployed metadata analytics value was produced by a different CDK CLI or library combination than the reviewed assembly.",
        "Do not hide the change; compare deployed and proposed templates, preserve the deployed metadata value in a corrected assembly if behavior is untouched, and create a new non-executed change set for review.",
        "Record caller account and Region, stack state, config fingerprint, template and asset hashes, change-set resource list, replacement flags, IAM scope, final status, and post-deploy metadata checks.",
        "reviewed assembly",
        "the exact synthesized template, asset manifest, configuration fingerprint, and toolchain evidence approved for an authenticated infrastructure action",
        "CloudFormation,CDK,change set,deployment,template hash,asset hash,replacement",
        "approval boundary|two-change example|metadata drift failure|deployment evidence checklist",
    ),
    _spec(
        "pipeline_retry_budget",
        "Retry Budgets for Event-Driven Pipelines",
        "AWS Lambda",
        "reliability",
        "operations_guide",
        "advanced",
        "limit the total repeated work an event may generate across S3 delivery, Lambda asynchronous retries, orchestration, and provider calls",
        "Assign one end-to-end attempt budget, make each layer expose its attempt number, and prevent nested retries from multiplying beyond the downstream recovery objective.",
        "An event permits one initial Lambda attempt and two bounded task retries for throttling, but a validation failure uses no provider retry and moves directly to a failed descriptor.",
        "A temporary provider throttle generates dozens of calls for one document and exhausts concurrency.",
        "Lambda, the workflow, and the provider client each retried independently without sharing an attempt budget or honoring retry-after behavior.",
        "Stop uncontrolled redrive, measure attempts by correlation ID, centralize retry ownership, cap exponential backoff, and verify one controlled transient-failure simulation offline.",
        "Count delivery attempts, function attempts, task attempts, provider attempts, elapsed retry time, DLQ movements, descriptor attempt fields, and duplicate suppressions.",
        "retry budget",
        "the maximum total attempts or elapsed retry time allowed for one logical operation across all participating layers",
        "retry,budget,Lambda,backoff,throttling,DLQ,attempt count",
        "end-to-end attempt model|bounded retry example|retry multiplication incident|attempt accounting",
    ),
    _spec(
        "manifest_state_reconciliation",
        "Manifest and Descriptor State Reconciliation",
        "Amazon S3",
        "reliability",
        "standard_operating_procedure",
        "advanced",
        "ensure document-level manifest state agrees with immutable per-chunk descriptors after retries, partial failures, or interrupted writes",
        "Treat chunk states as the detailed source of truth, validate their checksums, derive aggregate counts, and update the manifest through a conditional transition.",
        "For four chunks with three ready and one retryable failure, the descriptor reports indexed=3, failed=1, pending=0 and the manifest remains failed rather than ready.",
        "The manifest says ready while one chunk has no stored vector confirmation.",
        "An aggregate update used attempted chunk count instead of checksum-valid successful vector writes and was committed before the final write completed.",
        "Mark the document non-ready, compare descriptor records with vector-write confirmations, repair only missing chunks, recompute totals, and conditionally publish ready after all checks pass.",
        "Validate document and chunk checksums, unique indices, state totals, attempt and error fields, vector confirmation, descriptor version, manifest version, and transition history.",
        "ready state",
        "a terminal document state allowed only when every expected chunk has a validated embedding and confirmed deterministic vector write",
        "manifest,descriptor,state machine,chunk,checksum,vector,conditional write",
        "state source of truth|aggregate derivation|premature-ready incident|reconciliation checklist",
    ),
    _spec(
        "embedding_validation_gate",
        "Embedding Validation Before Vector Writes",
        "Amazon Bedrock",
        "data_quality",
        "security_policy",
        "advanced",
        "prevent malformed model output from entering the vector store or causing a document to be marked ready",
        "Require every embedding to be a non-empty numeric finite vector of the configured dimension before constructing any vector-store request.",
        "A configured dimension of 1024 accepts exactly 1024 finite floats; a vector containing NaN or 1023 values records a non-retryable validation failure and performs no write.",
        "Vector upsert errors appear intermittently after a model or configuration change.",
        "The runtime passed provider output directly to the vector client without verifying dimension and finite numeric values against the reviewed configuration.",
        "Disable writes for the affected model/configuration pair, validate persisted outputs, mark invalid chunks failed without exposing vector content, and restore only after offline contract tests pass.",
        "Test empty, boolean, string, NaN, infinity, wrong-dimension, and valid vectors; assert no vector call for failures and ready state only after confirmed valid writes.",
        "finite vector",
        "an embedding whose every element is a real numeric value other than NaN or positive or negative infinity",
        "embedding,validation,dimension,numeric,finite,Bedrock,vector write",
        "validation contract|dimension example|unchecked-output incident|negative vector tests",
    ),
    _spec(
        "qdrant_collection_safety",
        "Qdrant Collection and Payload Safety",
        "Qdrant",
        "vector_operations",
        "architecture_guide",
        "advanced",
        "write deterministic vectors to the authorized collection while preserving scope filters, dimension agreement, and secret handling",
        "Validate HTTPS endpoint and collection configuration offline, load the API key lazily from its secret reference, verify collection dimension at runtime, and upsert deterministic IDs with complete isolation payloads.",
        "A chunk point contains client, environment, namespace, domain, document_id, chunk_id, source, and checksum; retries upsert the same point rather than create another.",
        "A runtime discovers that the configured embedding dimension differs from the existing collection dimension.",
        "The embedding model changed without a versioned collection migration and the deployment validation checked only endpoint syntax.",
        "Perform no writes, mark the operation failed with a safe dimension error, review collection metadata without logging credentials, and plan a separately approved versioned migration.",
        "Verify credential-free HTTPS root URL, secret ARN rather than value, lazy client composition, collection name, dimensions, distance metric, deterministic IDs, payload scope, and no secret logging.",
        "collection contract",
        "the reviewed combination of collection identity, vector dimension, distance behavior, payload schema, and client/environment scope",
        "Qdrant,collection,payload,dimension,HTTPS,secret ARN,vector ID",
        "lazy secure composition|deterministic payload|dimension mismatch|collection contract evidence",
    ),
    _spec(
        "incident_dlq_triage",
        "Dead-Letter Queue Triage Without Message Loss",
        "Amazon SQS",
        "incident_response",
        "troubleshooting_runbook",
        "intermediate",
        "inspect failed asynchronous pipeline events without deleting historical evidence or triggering uncontrolled retries",
        "Use queue metadata and read-only message inspection under explicit authorization, distinguish historical from new failures by timestamps and correlation IDs, and redrive only a bounded reviewed set.",
        "A smoke test begins with three historical visible messages; after one successful event the count remains three, demonstrating that no new failure was added and no history was deleted.",
        "The visible message count falls during diagnosis even though no redrive was approved.",
        "A receive operation used a visibility timeout and a cleanup process or consumer removed messages, making count changes ambiguous.",
        "Stop all consumers, preserve queue metrics and CloudTrail evidence where available, wait for visibility restoration, identify any deletion, and do not redrive until scope is understood.",
        "Record approximate visible and in-flight counts, oldest age, receive/delete metrics, correlation IDs, timestamps, redrive policy, consumer state, and explicit evidence-retention decision.",
        "historical DLQ message",
        "a previously failed event retained as evidence whose presence must not be attributed to a new controlled test without timestamp and correlation proof",
        "SQS,DLQ,triage,visibility timeout,redrive,evidence,correlation",
        "metadata-first triage|historical-count example|visibility ambiguity|evidence-preserving verification",
    ),
    _spec(
        "deployment_rollback_readiness",
        "Deployment Rollback Readiness Checklist",
        "AWS CloudFormation",
        "deployment_safety",
        "deployment_checklist",
        "intermediate",
        "prepare a pipeline release so a failed update reaches a known stable state without data deletion or unreviewed retries",
        "Identify replacement and deletion risk, preserve compatible prior artifacts and configuration, define terminal stack states, and separate infrastructure rollback from data repair.",
        "Before enabling automatic indexing, operators retain the prior Lambda asset hash, verify no resource replacements, snapshot configuration fingerprints, and define how pending descriptors remain resumable.",
        "A failed Lambda update rolls back its code, but new descriptor records cannot be read by the prior runtime.",
        "The release changed persisted schema without backward-reading support or a staged migration boundary.",
        "Stop new events, preserve affected descriptors, deploy a compatible reader through a separately reviewed change, reconcile states, and only then resume ingestion.",
        "Review change set, replacement flags, removal policies, prior asset availability, schema compatibility, event pause procedure, stack terminal states, alarms, and data reconciliation owner.",
        "rollback boundary",
        "the exact infrastructure and persisted-data versions that can safely resume together after an unsuccessful release",
        "CloudFormation,rollback,deployment,schema compatibility,asset hash,resumability",
        "rollback plan|automatic-indexing example|schema incompatibility|readiness checklist",
    ),
)


def _summary(spec: DocumentSpec) -> str:
    return f"Operational guidance to {spec.objective}."


def _document_text(spec: DocumentSpec) -> str:
    keywords = ", ".join(spec.keywords)
    topics = ", ".join(spec.chunk_topics)
    return f"""# {spec.title}

> **Synthetic-data notice:** {SYNTHETIC_NOTICE}

## Purpose and operating context

This guide explains how to {spec.objective}. It is written for an operator who must preserve evidence, respect client and environment boundaries, and make a change only after its scope is understood. The recommended procedure assumes that raw inputs, derived artifacts, descriptors, and manifests have stable identities. It also assumes that retries can occur, so success is established from persisted state rather than from a single green invocation metric.

The focused control is: {spec.control} The control should be expressed in configuration, policy, or state-transition logic that can be reviewed offline. Record the client, environment, namespace, domain, document or partition identity, and configuration version whenever those dimensions apply. That evidence makes a later incident traceable without logging document content, credentials, tokens, connection strings, or secret values.

## Control design

Treat the operation as a sequence of bounded transitions. First identify the authoritative input and its immutable checksum or version. Next identify the exact output scope and the condition that permits promotion. Then determine which failures are retryable and which require quarantine or human review. A retry must reuse the same operation identity; it must not invent another document, partition, chunk, execution side effect, or vector point.

For this subject, **{spec.term}** means {spec.term_definition}. That definition matters because similar service terms can describe different evidence. An infrastructure status, event-delivery status, application descriptor, and business-data completeness signal are not interchangeable. Operators should cite the precise signal they used and should preserve a negative control that demonstrates adjacent client, environment, prefix, or state access remains denied where relevant.

## Standard procedure

1. **Freeze scope.** Write down the affected client, environment, data domain, time window, input identity, and current persisted state. Do not broaden the window merely to make a test convenient.
2. **Capture a baseline.** Record safe metadata, counts, checksums, state versions, alarms, and configuration fingerprints before changing or retrying anything. Historical DLQ messages and test artifacts are evidence, not cleanup targets.
3. **Apply the focused control.** {spec.control} Change one variable at a time and keep production behavior opt-in when the control enables external or costly work.
4. **Exercise the reviewed path.** Use a deterministic fixture or the smallest authorized production event. Never substitute repeated uploads, broad replay, or wildcard permission for diagnosis.
5. **Reconcile outputs.** {spec.verification} A document or partition is ready only when every required state and side effect is confirmed; absence of an error message is insufficient.

## Practical example

{spec.example} The example deliberately keeps names synthetic and omits account numbers and credentials. The same method should be applied with the deployment's authorized configuration rather than copying identifiers from this document. When a service is at-least-once, duplicate delivery is expected input behavior; deterministic IDs and conditional state transitions are what keep the resulting data correct.

An operator should keep an evidence row containing the operation ID, input checksum, expected outputs, observed outputs, attempt count, and final state. If the operation touches multiple services, use one correlation ID but retain each service's native request or execution identifier separately. This prevents a successful handoff from being mistaken for successful downstream processing.

## Failure mode and remediation

**Observed failure:** {spec.failure}

**Likely cause:** {spec.cause}

**Safe remediation:** {spec.remediation} Preserve the original inputs and failed state until reconciliation is complete. If a repair would create, replace, delete, replay, or invoke an external resource, stop at the review boundary and obtain the required approval. Do not retrieve secret payloads merely to prove that a secret reference exists; metadata and configuration validation should be used when the value is not needed.

A second common failure is declaring success from an aggregate count while one detailed item is missing. Compare the expected set with the actual set by deterministic identity, not only by total. A third failure is expanding IAM, a time window, or a namespace after AccessDenied or empty results. Trace the exact request and data path first; broadening scope can hide the cause and violate isolation.

## Monitoring, evidence, and warnings

The minimum operational evidence for this guide is: {spec.verification} Useful overlapping terms for retrieval and incident correlation include {keywords}. Expected chunk topics are {topics}. Metrics and logs must use safe identifiers and counts. They must not contain source record bodies, API keys, passwords, authorization headers, secret values, or customer contact information.

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
"""


def _metadata_record(spec: DocumentSpec) -> dict[str, Any]:
    return {
        "document_id": spec.document_id,
        "filename": spec.filename,
        "title": spec.title,
        "summary": _summary(spec),
        "domain": spec.domain,
        "service": spec.service,
        "document_type": spec.document_type,
        "difficulty": spec.difficulty,
        "keywords": list(spec.keywords),
        "synthetic": True,
        "license": LICENSE,
        "created_at": CREATED_AT,
        "expected_chunk_topics": list(spec.chunk_topics),
    }


def _query_records(spec: DocumentSpec) -> list[dict[str, Any]]:
    document_id = spec.document_id
    services = [value.strip() for value in spec.service.split(" and ")]
    negative_variants = (
        (
            f"Which real customer's support contact and negotiated vendor price "
            f"should be used to {spec.objective}?",
            "The synthetic corpus contains no real customer contacts or "
            "negotiated vendor prices, so the question cannot be answered.",
        ),
        (
            f"What exact current regional service price and private discount "
            f"applies when an operator needs to {spec.objective}?",
            "The corpus does not contain current regional prices or private "
            "discount agreements, so the question cannot be answered.",
        ),
        (
            f"What production account identifier and secret value should an "
            f"engineer use to {spec.objective}?",
            "The corpus does not contain production account identifiers or "
            "secret values, and those details must not be inferred.",
        ),
        (
            f"Which named incident participant and private phone number owns the "
            f"decision to {spec.objective}?",
            "The synthetic corpus contains no real incident participants or "
            "private phone numbers, so the question cannot be answered.",
        ),
    )
    negative_query, negative_summary = negative_variants[
        DOCUMENT_SPECS.index(spec) % len(negative_variants)
    ]
    common = {
        "relevant_document_ids": [document_id],
        "primary_document_id": document_id,
        "answerable": True,
        "difficulty": spec.difficulty,
        "services": services,
        "forbidden_document_ids": [],
    }
    return [
        {
            "query_id": f"rq-{document_id}-direct",
            "scenario_id": f"scenario-{document_id}-control",
            "query": f"Which primary control does the guide recommend to {spec.objective}?",
            "expected_answer_summary": spec.control,
            "query_type": "direct_fact",
            "required_keywords": list(spec.keywords[:3]),
            **common,
        },
        {
            "query_id": f"rq-{document_id}-paraphrase",
            "scenario_id": f"scenario-{document_id}-control",
            "query": f"How should a data engineer keep {spec.title.lower()} repeatable, reviewable, and correctly scoped?",
            "expected_answer_summary": (
                f"Use the focused control, deterministic identities, bounded scope, "
                f"and persisted evidence. {spec.control}"
            ),
            "query_type": "paraphrase",
            "required_keywords": [spec.keywords[0], "scope", "evidence"],
            **common,
        },
        {
            "query_id": f"rq-{document_id}-troubleshooting",
            "scenario_id": f"scenario-{document_id}-failure",
            "query": f"An operator observes this condition: {spec.failure} What should be checked first, and what is the safe remediation?",
            "expected_answer_summary": f"Check whether {spec.cause} Then {spec.remediation}",
            "query_type": "troubleshooting",
            "required_keywords": [spec.keywords[0], "cause", "remediation"],
            **common,
        },
        {
            "query_id": f"rq-{document_id}-terminology",
            "scenario_id": f"scenario-{document_id}-terminology",
            "query": f"In the context of {spec.title.lower()}, what does the term '{spec.term}' mean operationally?",
            "expected_answer_summary": spec.term_definition,
            "query_type": "ambiguous_terminology",
            "required_keywords": [spec.term, spec.keywords[0]],
            **common,
        },
        {
            "query_id": f"rq-{document_id}-multistep",
            "scenario_id": f"scenario-{document_id}-procedure",
            "query": f"What ordered checks should be completed before declaring success when trying to {spec.objective}?",
            "expected_answer_summary": (
                "Freeze scope, capture a safe baseline, apply the focused control, "
                f"exercise one bounded path, and reconcile outputs. {spec.verification}"
            ),
            "query_type": "multi_step",
            "required_keywords": ["scope", "baseline", "reconcile"],
            **common,
        },
        {
            "query_id": f"rq-{document_id}-unanswerable",
            "scenario_id": f"scenario-{document_id}-unanswerable",
            "query": negative_query,
            "relevant_document_ids": [],
            "primary_document_id": None,
            "expected_answer_summary": negative_summary,
            "answerable": False,
            "difficulty": "intermediate",
            "query_type": "unanswerable",
            "services": services,
            "required_keywords": [],
            "forbidden_document_ids": [document_id],
        },
    ]


def _answer_records(spec: DocumentSpec) -> list[dict[str, Any]]:
    document_id = spec.document_id
    return [
        {
            "case_id": f"ae-{document_id}-control",
            "scenario_id": f"scenario-{document_id}-control",
            "question": f"What control is central to {spec.title.lower()}?",
            "expected_answer": spec.control,
            "required_facts": [spec.control, spec.term_definition],
            "prohibited_claims": [
                "The guide authorizes wildcard access.",
                "A successful invocation alone proves end-to-end readiness.",
            ],
            "supporting_document_ids": [document_id],
            "citation_required": True,
            "answerable": True,
            "difficulty": spec.difficulty,
            "category": spec.domain,
        },
        {
            "case_id": f"ae-{document_id}-failure",
            "scenario_id": f"scenario-{document_id}-failure",
            "question": f"How should an operator respond when {spec.failure}",
            "expected_answer": f"The likely cause is that {spec.cause} The safe response is to {spec.remediation[0].lower() + spec.remediation[1:]}",
            "required_facts": [spec.cause, spec.remediation],
            "prohibited_claims": [
                "Delete historical evidence before diagnosis.",
                "Broaden permissions or replay scope without review.",
            ],
            "supporting_document_ids": [document_id],
            "citation_required": True,
            "answerable": True,
            "difficulty": "advanced",
            "category": "troubleshooting",
        },
        {
            "case_id": f"ae-{document_id}-procedure",
            "scenario_id": f"scenario-{document_id}-procedure",
            "question": f"Summarize the evidence-based completion sequence for {spec.title.lower()}.",
            "expected_answer": (
                "Freeze the exact scope, capture baseline metadata and checksums, "
                "apply the focused control, exercise one bounded path, and reconcile "
                f"every expected output before declaring readiness. {spec.verification}"
            ),
            "required_facts": [
                "Freeze the affected scope.",
                "Capture a safe baseline.",
                "Reconcile detailed outputs before readiness.",
                spec.verification,
            ],
            "prohibited_claims": [
                "Missing error logs are sufficient evidence of success.",
                "Retries may create new operation identities.",
            ],
            "supporting_document_ids": [document_id],
            "citation_required": True,
            "answerable": True,
            "difficulty": spec.difficulty,
            "category": "operational_procedure",
        },
    ]


def build_records() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    metadata = [_metadata_record(spec) for spec in DOCUMENT_SPECS]
    queries = [record for spec in DOCUMENT_SPECS for record in _query_records(spec)]
    answers = [record for spec in DOCUMENT_SPECS for record in _answer_records(spec)]
    return metadata, queries, answers


def _split_records(
    queries: list[dict[str, Any]], answers: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    ordered_ids = sorted(
        (spec.document_id for spec in DOCUMENT_SPECS),
        key=lambda value: hashlib.sha256(
            f"{SPLIT_SEED}|{value}".encode("utf-8")
        ).hexdigest(),
    )
    groups = {
        "train": ordered_ids[:25],
        "validation": ordered_ids[25:30],
        "test": ordered_ids[30:],
    }
    result: dict[str, dict[str, Any]] = {}
    for name, document_ids in groups.items():
        allowed = set(document_ids)
        query_ids = [
            record["query_id"]
            for record in queries
            if (
                record["primary_document_id"] in allowed
                or any(
                    forbidden in allowed
                    for forbidden in record["forbidden_document_ids"]
                )
            )
        ]
        answer_ids = [
            record["case_id"]
            for record in answers
            if set(record["supporting_document_ids"]) & allowed
        ]
        result[name] = {
            "dataset_version": DATASET_VERSION,
            "split": name,
            "seed": SPLIT_SEED,
            "strategy": (
                "SHA-256 ordering of document IDs; documents and every "
                "derived scenario, paraphrase, query, and answer case remain "
                "in one split"
            ),
            "document_ids": document_ids,
            "retrieval_query_ids": query_ids,
            "answer_case_ids": answer_ids,
            "counts": {
                "documents": len(document_ids),
                "retrieval_queries": len(query_ids),
                "answer_cases": len(answer_ids),
            },
        }
    return result


def _jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )


def render_dataset_files() -> dict[Path, str]:
    """Return every generated dataset file as deterministic UTF-8 text."""

    metadata, queries, answers = build_records()
    splits = _split_records(queries, answers)
    files = {
        DATASET_ROOT / "documents" / spec.filename: _document_text(spec)
        for spec in DOCUMENT_SPECS
    }
    files[DATASET_ROOT / "metadata/documents.jsonl"] = _jsonl(metadata)
    files[DATASET_ROOT / "evaluation/retrieval_queries.jsonl"] = _jsonl(
        queries
    )
    files[DATASET_ROOT / "evaluation/answer_evaluation.jsonl"] = _jsonl(
        answers
    )
    for name, payload in splits.items():
        files[DATASET_ROOT / f"splits/{name}.json"] = (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
    return dict(sorted(files.items(), key=lambda item: item[0].as_posix()))


def write_dataset(repository_root: Path = REPOSITORY_ROOT) -> None:
    for relative_path, content in render_dataset_files().items():
        destination = repository_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8", newline="\n")


def check_dataset(repository_root: Path = REPOSITORY_ROOT) -> list[str]:
    errors: list[str] = []
    for relative_path, content in render_dataset_files().items():
        destination = repository_root / relative_path
        if not destination.is_file():
            errors.append(f"missing generated file: {relative_path.as_posix()}")
        elif destination.read_text(encoding="utf-8") != content:
            errors.append(f"generated file differs: {relative_path.as_posix()}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify static files match deterministic generation",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
    )
    arguments = parser.parse_args()
    root = arguments.repository_root.resolve()
    if arguments.check:
        errors = check_dataset(root)
        if errors:
            for error in errors:
                print(f"ERROR: {error}")
            return 1
        print(f"Dataset is deterministic ({len(render_dataset_files())} files).")
        return 0
    write_dataset(root)
    print(f"Generated {len(render_dataset_files())} dataset files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
