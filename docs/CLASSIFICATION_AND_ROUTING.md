# Classification and request routing

The classification and routing layer determines how a request should be
handled. It does not answer the request, retrieve documents, invoke tools, or
change AWS resources. It is application code and adds no CDK infrastructure.

## Flow

```text
User request + optional context
              |
              v
  IntentClassifier protocol
              |
              v
     ClassificationResult
              |
              v
        RequestRouter
              |
              v
         RoutingPlan
              |
      +-------+--------+------------------+
      |       |        |                  |
   direct  retrieval  code/tools   approval or safety
 response   planning   planning        review
```

The router creates a plan only. A future orchestrator must separately decide
whether and how to perform retrieval or invoke a tool.

## Classifier and embedder boundaries

`IntentClassifier` and `EmbeddingProvider` are intentionally separate
protocols:

- A classifier maps a request to a handling intent and safety flags.
- An embedder maps text to numeric vectors for semantic retrieval.
- Classification never returns vectors.
- Embedding never authorizes actions or chooses a route.

`RuleBasedIntentClassifier` is the current implementation. It is deterministic,
network-free, and safe to use in unit tests. `BedrockIntentClassifier` is an
explicit placeholder and raises `NotImplementedError`; it does not construct a
client or make an AWS request.

## Classification schema

`ClassificationResult` is an immutable typed record:

```json
{
  "intent": "pipeline_troubleshooting",
  "confidence": 0.98,
  "requires_retrieval": true,
  "requires_tool_call": false,
  "requires_approval": false,
  "preferred_knowledge_scope": "client_environment",
  "reasoning_summary": "The request describes a pipeline problem to diagnose.",
  "matched_rules": ["pipeline_troubleshooting"]
}
```

`reasoning_summary` is a fixed, short, user-safe explanation. It is not hidden
reasoning or chain-of-thought. Complete request text is not included in
structured logs.

Supported intents are:

- `architecture_design`
- `pipeline_requirements`
- `pipeline_generation`
- `pipeline_troubleshooting`
- `aws_error_explanation`
- `sql_generation`
- `pyspark_generation`
- `cdk_generation`
- `iam_review`
- `data_quality`
- `monitoring_request`
- `cost_question`
- `knowledge_question`
- `deployment_request`
- `destructive_action_request`
- `general_conversation`
- `unknown`

## Deterministic rules and priority

Rules contain a stable name, intent, priority, phrases, keywords, minimum
keyword-match count, and downstream hints. Text is normalized using Unicode
normalization, case folding, punctuation removal, and whitespace collapsing.

The classifier:

1. Checks execution-oriented safety rules first.
2. Scores all matching intent rules.
3. Selects the strongest match and uses explicit priority as a deterministic
   tie-breaker.
4. Lowers confidence and reports both rule names when two similarly weighted
   rules are ambiguous.
5. Returns `unknown` when no rule clears the configured confidence threshold.

Rules are constructor-injected, so keyword and phrase sets can be extended
without changing the provider contract.

Examples:

| Request | Intent | Route |
| --- | --- | --- |
| `My Glue job failed` | `pipeline_troubleshooting` | `troubleshooting` |
| `Build PostgreSQL to S3 pipeline` | `pipeline_generation` | `code_generation` |
| `Write a PySpark transformation` | `pyspark_generation` | `code_generation` |
| `Explain this IAM access denied error` | `aws_error_explanation` | `troubleshooting` |
| `Review this IAM policy` | `iam_review` | `retrieval` |
| `Check current alarm status` | `monitoring_request` | `tool_execution` |

## Routing schema

`RoutingPlan` preserves:

- selected route and intent
- retrieval requirement, top-k, and knowledge scope
- tool category
- approval and safety-review requirements
- a short next action
- classifier confidence
- client ID and environment

Available routes are `direct_response`, `retrieval`,
`requirements_gathering`, `code_generation`, `troubleshooting`,
`tool_execution`, `approval_required`, and
`rejection_or_safety_review`.

The `tool_execution` route is still only a plan. It does not authorize or
perform a tool call.

## Approval and safety

Deployment and destructive actions always require explicit approval. The
router enforces this from validated configuration even if a future classifier
incorrectly returns `requires_approval=false`.

Destructive actions also require safety review. Current rules cover requests to
deploy, destroy, delete, replace, modify production, broaden IAM permissions,
move data, or overwrite data. Discussing or explaining one of these operations
is not permission to perform it:

| Request | Result |
| --- | --- |
| `Deploy the CDK stack` | Approval required |
| `Delete the production bucket` | Approval and safety review required |
| `Explain why deleting a production bucket is dangerous` | Retrieval; no action authorization |
| `How would a CDK deployment work?` | Retrieval; no action authorization |

An execution layer must still validate exact targets, permissions, and current
user approval immediately before any mutation.

## Multi-client isolation

The classifier accepts optional `client_id` and `environment` context without
using values from one client to classify another. The router copies these
values unchanged into the routing plan. Retrieval defaults to the
`client_environment` scope so a downstream retriever can enforce the same
boundary.

This layer does not create global caches or merge conversation, metadata, or
knowledge across clients. A future orchestrator must use both fields as hard
filters when loading history or retrieving documents.

## Configuration

`ClassificationRoutingConfig` validates:

- minimum classifier confidence
- unknown threshold
- default retrieval top-k
- approval-required intents
- safety-review intents
- classifier version

The configuration cannot remove mandatory approval from deployment or
destructive actions, or remove safety review from destructive actions.

## Structured logging

Routing emits one JSON event containing classifier version, predicted intent,
confidence, route, client ID, environment, and elapsed time. It never logs the
complete query, credentials, full documents, embeddings, or private reasoning.
Production log access and retention remain governed by the existing
infrastructure.

## Evaluation

`ClassificationEvaluator` evaluates labeled cases and returns:

- expected and predicted intent sequences
- accuracy
- precision, recall, F1, and support per intent
- confusion-matrix counts
- predicted unknown rate

The included representative synthetic dataset contains every supported intent.
It is deliberately small and is a test foundation, not a production quality
claim. Future curated datasets should include redacted real-world phrasing,
class imbalance, multi-turn context, adversarial safety wording, and
client-specific vocabulary.

## Future Bedrock classifier

`BedrockIntentClassifier` reserves the provider boundary but contains no
Bedrock integration. TODO markers identify the deferred work:

- schema-constrained structured JSON output
- prompt versioning
- confidence calibration
- deterministic rule fallback
- labeled provider evaluation

Before enabling the provider, add mocked unit tests, explicit timeout and error
handling, token and cost budgets, IAM least privilege, redaction controls, and
an offline comparison against the deterministic baseline. No Bedrock
classifier permissions or resources are needed now.
