# Retrieval strategy evaluation

This evaluation compares the repository's local semantic-vector path with a
new BM25 keyword retriever and reciprocal-rank hybrid retriever. It runs
entirely offline, uses no AWS credentials, and provisions no infrastructure or
managed vector store.

## Benchmark design

The versioned benchmark is
[`evaluation/benchmark/retrieval_benchmark.json`](../evaluation/benchmark/retrieval_benchmark.json).
Its licensing boundary is documented in
[`evaluation/benchmark/README.md`](../evaluation/benchmark/README.md).
It contains 35 synthetic queries against the seven CC0 demo documents:

| Dimension | Composition |
| --- | --- |
| Subject areas | S3 architecture, Glue troubleshooting, Athena troubleshooting, IAM least privilege, PySpark transformations, monitoring, and cost awareness |
| Queries per subject | 5 |
| Match type | 14 exact-keyword, 14 paraphrase, 7 ambiguous |
| Difficulty | 14 easy, 11 medium, 10 hard |
| Scope | `demo-client` / `dev` |
| Labels | Expected document IDs and optional expected chunk IDs |

Every case has a stable query ID, query, category, difficulty, match type,
notes, and at least one expected document. The loader rejects malformed JSON,
duplicate IDs, missing coverage, unsupported values, or fewer than 25 cases.
The runner also verifies that every expected document and chunk exists in the
current corpus.

The benchmark contains no commercial book text, LLM Zoomcamp FAQ content,
private records, or customer data. The queries were written for this project
and share the demo corpus's CC0 boundary.

## Retrieval strategies

### Semantic-vector baseline

`InMemoryCosineRetriever` is unchanged. The comparison embeds the corpus and
queries with the current offline `DeterministicDemoEmbeddingProvider`, then
ranks finite, same-dimension vectors by cosine similarity.

This is the application's current demo vector path, but its token-hash
embeddings are not a pretrained language model. Calling it the semantic-vector
baseline describes the retrieval architecture, not production-grade semantic
understanding. A future evaluation with a reviewed Bedrock or local embedding
model is needed before generalizing these results.

### BM25 keyword retrieval

`InMemoryBM25Retriever` implements the standard BM25 term-frequency and
inverse-document-frequency ranking locally. Defaults are:

- `k1 = 1.5`
- `b = 0.75`
- `top_k = 5`
- `minimum_score = 0.0`

Tokenization is deterministic, case-insensitive, and limited to ASCII
alphanumeric tokens. The implementation adds no package dependency and makes
no network call. A zero score is treated as no lexical match. Ties sort by
document ID and then chunk ID.

Every query requires an explicit client ID and environment. Entries without an
exact metadata match are excluded before document-frequency statistics or
ranking are calculated.

### Reciprocal-rank hybrid retrieval

`ReciprocalRankFusionRetriever` constructs semantic and BM25 rankings only from
entries in the requested client/environment scope. It removes duplicate
document/chunk pairs and combines ranks using:

```text
RRF(d) = semantic_weight / (rank_constant + semantic_rank)
       + keyword_weight  / (rank_constant + keyword_rank)
```

The evaluated configuration uses equal weights, rank constant 60, and a
candidate pool of 50. Reciprocal rank fusion is used because cosine and BM25
scores have different scales. Fusion settings are explicit and reject zero,
negative, non-finite, or otherwise invalid values.

## Metrics

Metrics are calculated independently for each strategy:

- Precision@1, @3, and @5: relevant results divided by k.
- Recall@1, @3, and @5: unique expected targets found divided by all expected
  targets.
- MRR: mean inverse rank of the first relevant result.
- Hit rate: share of cases with a relevant result in the top five.
- No-result rate: share of cases returning an empty list.
- Exact-document success: share of cases where every expected document appears
  in the top five.
- Category, difficulty, and match-type performance: grouped hit rate, MRR, and
  recall@5.
- Average, p50, and nearest-rank p95 latency: calculated from per-query median
  wall-clock latency over five repetitions.

Latency includes query embedding for semantic and hybrid strategies. It is a
local development measurement, not a throughput or service-level benchmark.

## Reproduce the comparison

From the repository root in Windows PowerShell:

```powershell
python -m evaluation.run_retrieval_comparison
```

The command writes:

- [`evaluation/results/retrieval_comparison.json`](../evaluation/results/retrieval_comparison.json)
- [`evaluation/results/retrieval_comparison.md`](../evaluation/results/retrieval_comparison.md)
- [`evaluation/results/retrieval_query_results.csv`](../evaluation/results/retrieval_query_results.csv)

Override paths or settings through the CLI:

```powershell
python -m evaluation.run_retrieval_comparison --help
python -m evaluation.run_retrieval_comparison --rrf-rank-constant 40 --semantic-weight 1.25 --keyword-weight 1.0
```

Malformed benchmark data or invalid configuration returns exit code 2. Ranking
and quality metrics are deterministic for the versioned inputs. Evaluation
timestamps and measured latency legitimately vary by run and machine.

The reviewed result snapshot is committed because it is submission evidence.
Regenerate and review all three files together whenever the corpus, benchmark,
tokenizer, embedding provider, retrieval settings, or metrics change.

## Reviewed results

Snapshot provenance:

- Evaluation timestamp: `2026-07-28T01:33:03Z`
- Application version: `retrieval-evaluation-v1`
- Benchmark version: `retrieval-benchmark-v1`
- Corpus version: `synthetic-demo-corpus-v1`
- Corpus checksum:
  `sha256:a1c2a62ccfb4192dfc4caef6b03806a3104045efcc067fb47a25f25b9f688d78`
- Python: `3.12.10`
- Embedding model: `deterministic-demo-keyword-v1`

| Strategy | P@1 | P@3 | P@5 | R@1 | R@3 | R@5 | MRR | Hit@5 | Exact doc | No result | P95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Semantic vector | 0.4000 | 0.2476 | 0.1771 | 0.4000 | 0.7429 | 0.8857 | 0.5948 | 0.8857 | 0.8857 | 0.0000 | 0.078 ms |
| BM25 keyword | 0.8571 | 0.3333 | 0.2000 | 0.8571 | 1.0000 | 1.0000 | 0.9190 | 1.0000 | 1.0000 | 0.0000 | 0.144 ms |
| RRF hybrid | 0.6857 | 0.3143 | 0.2000 | 0.6857 | 0.9429 | 1.0000 | 0.8129 | 1.0000 | 1.0000 | 0.0000 | 0.289 ms |

BM25 was best overall, on exact-keyword queries, and on the current paraphrase
set. Its exact-keyword MRR was 1.0000 and paraphrase MRR was 0.8810. Hybrid
recovered all expected documents by k=5 but diluted several strong lexical
rankings with weaker vector ranks. It also performed both component searches,
making it the slowest strategy. Semantic vector retrieval was the fastest but
missed four expected documents at k=5:

- `s3-ambiguous-landing`
- `glue-paraphrase-input`
- `athena-paraphrase-columnar`
- `cost-paraphrase-safe-savings`

The corpus is small enough that all measured latencies are sub-millisecond;
their ordering is more meaningful than their absolute values.

## Strategy decision

The selection score is:

```text
0.5 * MRR + 0.3 * hit_rate + 0.2 * recall@3
```

MRR, recall@1, and no-result rate are deterministic tie-breakers. An exact tie
preserves the existing semantic default for backward compatibility.

For this synthetic offline corpus, **BM25 keyword retrieval is the recommended
demo default**. The application default has not been changed. A Streamlit
selector is also deferred: wiring it now would require changing the
vector-oriented application retrieval contract and response metadata before a
production semantic model has been evaluated. That UI churn is not justified
by this small-corpus benchmark. The comparison artifacts explicitly record the
strategy for every query.

A future backward-compatible integration should add a strategy-aware adapter at
the application boundary, keep mandatory client/environment filtering, expose
the selected strategy in typed retrieval metadata, and initially limit the
selector to demo mode.

## Limitations and next evaluation work

- The seven-document corpus is intentionally small and topically distinct.
- Labels were authored with the benchmark and have not received independent
  relevance review.
- Paraphrases still share some terms with their source documents, which favors
  BM25.
- The offline vector provider is deterministic but not a pretrained semantic
  embedding model.
- No spelling, stemming, phrase, synonym, or field-weighting enhancement is
  applied to BM25.
- No reranker is evaluated.
- Latency excludes network services, persistent indexes, concurrency, and
  large-corpus effects.

Next work should expand blinded relevance judgments, add hard negatives, test a
reviewed real semantic model without committing sensitive data, compare
weighted RRF settings on a held-out set, and evaluate quality and latency as the
corpus grows.

## Zoomcamp evidence

The committed benchmark, runner, machine-readable results, per-query CSV,
reviewer report, configuration, failure list, and tests provide reproducible
retrieval-evaluation evidence. They support the retrieval-search and evaluation
parts of the rubric without claiming that the current synthetic benchmark
represents production performance.

## AWS pipeline operations test split

The repository also evaluates the independent 36-document CC BY 4.0 corpus.
Its leakage-safe test split contains six unseen documents, 30 scored answerable
queries, and six separately recorded unanswerable queries. The reviewed result
is:

| Strategy | MRR | Hit@5 | Recall@5 |
| --- | ---: | ---: | ---: |
| Semantic vector | 0.3372 | 0.5000 | 0.5000 |
| BM25 keyword | 1.0000 | 1.0000 | 1.0000 |
| RRF hybrid | 0.7306 | 0.9333 | 0.9333 |

Run `python -m evaluation.run_aws_pipeline_operations_evaluation --split test`
and review
[`evaluation/results/aws_pipeline_operations/evaluation_summary.md`](../evaluation/results/aws_pipeline_operations/evaluation_summary.md).
The unanswerable cases are not included in P@k, R@k, or MRR; no abstention
quality is claimed for this snapshot.
