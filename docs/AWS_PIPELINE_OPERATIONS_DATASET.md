# AWS Data Pipeline Operations Knowledge Base

## Purpose

The AWS Data Pipeline Operations Knowledge Base is a deterministic, publicly
shareable synthetic corpus for exercising this repository's document ingestion,
chunking, retrieval, citation, and grounded-answer evaluation paths. It covers
AWS data-engineering design, operations, security, incident response, data
quality, cost, isolation, and deployment safety. It is an operational document
corpus, not an FAQ corpus, and it does not reuse the DataTalks.Club Zoomcamp FAQ.

Every source document states that it is synthetic and is not official AWS
documentation. Service names identify the technical subject; they do not imply
endorsement or official guidance. The content contains no real customer names,
accounts, contacts, credentials, or proprietary incident data.

## Folder structure

```text
data/aws_pipeline_operations/
  documents/                 36 UTF-8 Markdown source documents
  metadata/documents.jsonl   one metadata record per document
  evaluation/
    retrieval_queries.jsonl  216 retrieval-label records
    answer_evaluation.jsonl  108 grounded-answer cases
  splits/
    train.json
    validation.json
    test.json
scripts/
  generate_aws_pipeline_operations_dataset.py
  validate_aws_pipeline_operations_dataset.py
tests/unit/
  test_aws_pipeline_operations_dataset.py
```

The generator is the deterministic source of the static data files. It uses no
model, provider, AWS SDK, HTTP client, random runtime state, or current clock.
Regeneration therefore produces byte-identical UTF-8 files.

## Content and licensing rules

The 36 documents are original synthetic works focused on one operational
subject each. Documents are 500-1,500 words and include a practical example,
an explicit warning, a failure mode, a likely cause, remediation, verification
evidence, and a completion checklist. Shared terminology such as deterministic
identity, client/environment scope, descriptors, manifests, retries, checksums,
and readiness creates meaningful retrieval competition without duplicating
questions or source documents.

The dataset content is licensed under CC BY 4.0, as recorded in every metadata
record. Repository code remains under the repository's top-level license. The
fixed `created_at` value is dataset provenance, not a claim that the documents
were authored by AWS or observed in a real customer environment.

## Schemas

`metadata/documents.jsonl` uses stable `apo-NNN-slug` document IDs and contains:

- `document_id`, `filename`, `title`, `summary`
- `domain`, `service`, `document_type`, `difficulty`
- `keywords`, `expected_chunk_topics`
- `synthetic`, `license`, `created_at`

`evaluation/retrieval_queries.jsonl` contains six query styles per document:
direct fact, paraphrase, troubleshooting, ambiguous terminology, multi-step,
and deliberately unanswerable. Each record contains:

- `query_id`, `scenario_id`, `query`
- `relevant_document_ids`, `primary_document_id`
- `expected_answer_summary`, `answerable`
- `difficulty`, `query_type`, `services`
- `required_keywords`, `forbidden_document_ids`

For an unanswerable record, `primary_document_id` is `null`, relevance is empty,
and the expected summary says why the corpus cannot support an answer. Its
`forbidden_document_ids` field records the document that generated the negative
scenario; it is used for split grouping, not as evidence.

`evaluation/answer_evaluation.jsonl` contains:

- `case_id`, `scenario_id`, `question`, `expected_answer`
- `required_facts`, `prohibited_claims`, `supporting_document_ids`
- `citation_required`, `answerable`, `difficulty`, `category`

Expected answers are concise statements copied from or directly composed from
the associated synthetic source. Prohibited claims express unsafe or
unsupported conclusions that a grader should reject.

## Evaluation methodology

Retrieval evaluation should rank chunks and aggregate their document IDs before
computing document relevance. The repository's `RetrievalEvaluationCase` model
can directly represent the answerable records:

```python
import json
from pathlib import Path

from knowledge.evaluation import RetrievalEvaluationCase

records = [
    json.loads(line)
    for line in Path(
        "data/aws_pipeline_operations/evaluation/retrieval_queries.jsonl"
    ).read_text(encoding="utf-8").splitlines()
]
cases = [
    RetrievalEvaluationCase(
        query=record["query"],
        expected_document_ids=frozenset(record["relevant_document_ids"]),
    )
    for record in records
    if record["answerable"]
]
```

The existing model intentionally rejects a case with no relevant target, so
unanswerable records should be scored separately for abstention rather than
forced into precision/recall. For answer evaluation, check required facts,
prohibited claims, citations to supporting document IDs, and refusal behavior.
Exact string matching alone is not recommended.

## Deterministic splits and leakage prevention

The fixed seed is `aws-pipeline-operations-v1-fixed-seed`. The generator orders
document IDs by SHA-256 of the seed and ID, then assigns complete document
groups to train, validation, and test. Every query, paraphrase, base scenario,
and answer case derived from one document follows that document. No scenario or
document can appear in more than one split.

| Split | Documents | Retrieval queries | Answer cases |
| --- | ---: | ---: | ---: |
| Train | 25 | 150 | 75 |
| Validation | 5 | 30 | 15 |
| Test | 6 | 36 | 18 |

The test set should remain untouched while tuning retrieval. Because documents
are grouped rather than randomly distributing individual questions, test
performance measures transfer to unseen operational subjects more strictly than
a paraphrase-level split would.

## Validation and tests

Run the offline validator and focused tests from the repository root:

```powershell
python -m scripts.validate_aws_pipeline_operations_dataset
python -m pytest tests/unit/test_aws_pipeline_operations_dataset.py -q
```

To intentionally regenerate the static files or prove that they are current:

```powershell
python -m scripts.generate_aws_pipeline_operations_dataset
python -m scripts.generate_aws_pipeline_operations_dataset --check
```

## Offline evaluation runner

The dataset is connected to the repository's existing semantic, BM25, hybrid,
and prompt-contract evaluators. Run the untouched test split with:

```powershell
python -m evaluation.run_aws_pipeline_operations_evaluation --split test
```

The runner evaluates 30 answerable retrieval queries, records the six
unanswerable queries separately, and evaluates 18 fixed-evidence answer cases.
It writes JSON, Markdown, and CSV artifacts beneath
`evaluation/results/aws_pipeline_operations/`. The reviewed test snapshot
selects BM25 for retrieval and grounded evidence-first for prompt-contract
adherence. Answer scoring uses a deterministic fake provider and therefore
does not measure real language-model quality.

Do not tune using the test split. Use `--split train` or `--split validation`
for experiments, and keep generated exploratory output outside the committed
reviewed-results directory.

The validator checks schemas, exact references, document length and notices,
minimum counts, answerability rules, unique IDs, duplicate and near-duplicate
questions, split coverage and leakage, deterministic file content, JSON/JSONL,
and patterns associated with credentials, account IDs, private keys, tokens,
email addresses, or assigned secret values. It cannot prove the absence of
every conceivable identifying phrase, so catalog review remains part of a
public release.

## Local ingestion without providers

Markdown is already supported by `KnowledgeIngestionPipeline`. This example
creates canonical raw, processed, chunk, descriptor, metadata, and manifest
artifacts under an ignored local directory. It does not embed, index, connect
to Qdrant, or call AWS:

```python
from pathlib import Path

from knowledge.ingestion import KnowledgeIngestionPipeline
from knowledge.storage import FileSystemKnowledgeStorage

source = Path(
    "data/aws_pipeline_operations/documents/s3_prefix_isolation.md"
)
pipeline = KnowledgeIngestionPipeline(
    FileSystemKnowledgeStorage(".local/aws-pipeline-operations")
)
entry = pipeline.ingest(
    filename=source.name,
    content=source.read_bytes(),
    source=f"dataset://{source.name}",
)
print(entry.document_id, entry.chunk_count, entry.embedding_status)
```

Do not use the opt-in provider indexing script merely to validate this corpus;
that script is designed to connect to configured local providers. Embedding and
vector-store evaluation should be a separately authorized activity.

## Limitations

- The corpus is synthetic and cannot reproduce every service limit, regional
  feature, account policy, or production incident.
- It is not a substitute for current official AWS documentation or a security
  review. Service behavior and quotas can change.
- Relevance labels are primarily single-document labels. Overlapping language
  intentionally creates plausible secondary matches, but only fully supporting
  documents are labeled relevant.
- The corpus has no real customer data, provider output, embeddings, or vector
  collection. Baseline scores depend on the selected chunker and retriever.
- Negative cases test abstention on absent customer, commercial, credential,
  and private-incident facts; they do not exhaust every unanswerable category.

These limits are deliberate. They keep the dataset reproducible, safe to share,
and suitable for demonstrating a complete non-FAQ RAG workflow without network
or infrastructure dependencies.
