# Retrieval comparison results

This snapshot was generated offline from a repository-owned synthetic corpus (`CC-BY-4.0`).

## Provenance

- Evaluation date: `2026-08-06T00:00:00Z`
- Application version: `retrieval-evaluation-v1`
- Benchmark version: `aws-pipeline-operations-retrieval-test-v1`
- Corpus version: `aws-pipeline-operations-1.0.0`
- Corpus checksum: `sha256:6fb774f85e62578e349e0108deb308fe42c6ca147a50639e6268a2d8016b4802`
- Corpus documents/chunks: `36` / `288`
- Embedding provider: `deterministic-demo`
- Embedding model: `deterministic-demo-keyword-v1`
- Python version: `3.12.13`
- Scope: `aws-pipeline-operations-evaluation` / `test`

## Overall metrics

| Strategy | P@1 | P@3 | P@5 | R@1 | R@3 | R@5 | MRR | Hit rate | Exact doc | No result | Avg ms | P50 ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| semantic | 0.2333 | 0.1667 | 0.1133 | 0.2333 | 0.4333 | 0.5000 | 0.3372 | 0.5000 | 0.5000 | 0.0000 | 2.280 | 2.285 | 2.369 |
| keyword | 1.0000 | 0.7000 | 0.6000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 11.812 | 11.850 | 12.993 |
| hybrid | 0.6000 | 0.4000 | 0.3333 | 0.6000 | 0.8333 | 0.9333 | 0.7306 | 0.9333 | 0.9333 | 0.0000 | 16.317 | 16.339 | 17.264 |

## Selection

- Recommended default: **keyword**
- Best exact-keyword strategy: **keyword**
- Best paraphrase strategy: **keyword**
- Formula: 0.5 * MRR + 0.3 * hit_rate + 0.2 * recall@3; then MRR, recall@1, and no-result rate
- Tie policy: Exact metric ties preserve the existing semantic default, then keyword, then hybrid.

This recommendation does not silently change the Streamlit application default. The existing semantic path remains in place until a separate backward-compatible integration decision.

## Performance by match type

| Strategy | Match type | Cases | Hit rate | MRR | Recall@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| semantic | ambiguous | 18 | 0.5000 | 0.3352 | 0.5000 |
| semantic | exact_keyword | 6 | 0.6667 | 0.4306 | 0.6667 |
| semantic | paraphrase | 6 | 0.3333 | 0.2500 | 0.3333 |
| keyword | ambiguous | 18 | 1.0000 | 1.0000 | 1.0000 |
| keyword | exact_keyword | 6 | 1.0000 | 1.0000 | 1.0000 |
| keyword | paraphrase | 6 | 1.0000 | 1.0000 | 1.0000 |
| hybrid | ambiguous | 18 | 0.9444 | 0.7778 | 0.9444 |
| hybrid | exact_keyword | 6 | 0.8333 | 0.7500 | 0.8333 |
| hybrid | paraphrase | 6 | 1.0000 | 0.5694 | 1.0000 |

## Performance by category

| Strategy | Category | Cases | Hit rate | MRR | Recall@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| semantic | cost_optimization | 5 | 0.4000 | 0.1500 | 0.4000 |
| semantic | data_quality | 5 | 0.6000 | 0.3667 | 0.6000 |
| semantic | reliability | 5 | 0.4000 | 0.1400 | 0.4000 |
| semantic | schema_evolution | 5 | 0.8000 | 0.5667 | 0.8000 |
| semantic | security | 5 | 0.2000 | 0.2000 | 0.2000 |
| semantic | vector_operations | 5 | 0.6000 | 0.6000 | 0.6000 |
| keyword | cost_optimization | 5 | 1.0000 | 1.0000 | 1.0000 |
| keyword | data_quality | 5 | 1.0000 | 1.0000 | 1.0000 |
| keyword | reliability | 5 | 1.0000 | 1.0000 | 1.0000 |
| keyword | schema_evolution | 5 | 1.0000 | 1.0000 | 1.0000 |
| keyword | security | 5 | 1.0000 | 1.0000 | 1.0000 |
| keyword | vector_operations | 5 | 1.0000 | 1.0000 | 1.0000 |
| hybrid | cost_optimization | 5 | 0.8000 | 0.4167 | 0.8000 |
| hybrid | data_quality | 5 | 1.0000 | 0.8000 | 1.0000 |
| hybrid | reliability | 5 | 1.0000 | 0.8000 | 1.0000 |
| hybrid | schema_evolution | 5 | 1.0000 | 0.8500 | 1.0000 |
| hybrid | security | 5 | 0.8000 | 0.6500 | 0.8000 |
| hybrid | vector_operations | 5 | 1.0000 | 0.8667 | 1.0000 |

## Performance by difficulty

| Strategy | Difficulty | Cases | Hit rate | MRR | Recall@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| semantic | advanced | 20 | 0.6000 | 0.4183 | 0.6000 |
| semantic | intermediate | 10 | 0.3000 | 0.1750 | 0.3000 |
| keyword | advanced | 20 | 1.0000 | 1.0000 | 1.0000 |
| keyword | intermediate | 10 | 1.0000 | 1.0000 | 1.0000 |
| hybrid | advanced | 20 | 1.0000 | 0.8292 | 1.0000 |
| hybrid | intermediate | 10 | 0.8000 | 0.5333 | 0.8000 |

## Failure summary

### semantic

- Missed expected targets at k=5: rq-apo-008-athena-query-cost-guardrails-paraphrase, rq-apo-008-athena-query-cost-guardrails-terminology, rq-apo-008-athena-query-cost-guardrails-multistep, rq-apo-009-athena-result-location-security-direct, rq-apo-009-athena-result-location-security-paraphrase, rq-apo-009-athena-result-location-security-terminology, rq-apo-009-athena-result-location-security-multistep, rq-apo-011-lambda-concurrency-backpressure-direct, rq-apo-011-lambda-concurrency-backpressure-paraphrase, rq-apo-011-lambda-concurrency-backpressure-terminology, rq-apo-024-schema-registry-compatibility-multistep, rq-apo-033-embedding-validation-gate-terminology, rq-apo-033-embedding-validation-gate-multistep, rq-apo-034-qdrant-collection-safety-paraphrase, rq-apo-034-qdrant-collection-safety-troubleshooting
- Returned no results: none

### keyword

- Missed expected targets at k=5: none
- Returned no results: none

### hybrid

- Missed expected targets at k=5: rq-apo-008-athena-query-cost-guardrails-direct, rq-apo-009-athena-result-location-security-multistep
- Returned no results: none

## Settings

```json
{
  "bm25": {
    "b": 0.75,
    "k1": 1.5
  },
  "evaluated_k_values": [
    1,
    3,
    5
  ],
  "fusion": {
    "candidate_pool_size": 50,
    "keyword_weight": 1.0,
    "method": "reciprocal_rank_fusion",
    "rank_constant": 60,
    "semantic_weight": 1.0
  },
  "hybrid_minimum_score": 0.0,
  "keyword_minimum_score": 0.0,
  "latency_repetitions": 5,
  "scope": {
    "client_id": "aws-pipeline-operations-evaluation",
    "environment": "test"
  },
  "semantic_minimum_similarity": 0.0,
  "top_k": 5
}
```

Latency is local wall-clock time and is environment-dependent. Each reported query latency is the median of the configured repetitions. Ranking and quality metrics are deterministic for the versioned corpus, benchmark, and settings.
