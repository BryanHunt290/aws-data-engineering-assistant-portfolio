# Retrieval comparison results

This snapshot was generated offline from the repository-owned synthetic CC0 corpus.

## Provenance

- Evaluation date: `2026-07-28T01:33:03Z`
- Application version: `retrieval-evaluation-v1`
- Benchmark version: `retrieval-benchmark-v1`
- Corpus version: `synthetic-demo-corpus-v1`
- Corpus checksum: `sha256:a1c2a62ccfb4192dfc4caef6b03806a3104045efcc067fb47a25f25b9f688d78`
- Corpus documents/chunks: `7` / `7`
- Embedding provider: `deterministic-demo`
- Embedding model: `deterministic-demo-keyword-v1`
- Python version: `3.12.10`
- Scope: `demo-client` / `dev`

## Overall metrics

| Strategy | P@1 | P@3 | P@5 | R@1 | R@3 | R@5 | MRR | Hit rate | Exact doc | No result | Avg ms | P50 ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| semantic | 0.4000 | 0.2476 | 0.1771 | 0.4000 | 0.7429 | 0.8857 | 0.5948 | 0.8857 | 0.8857 | 0.0000 | 0.074 | 0.074 | 0.078 |
| keyword | 0.8571 | 0.3333 | 0.2000 | 0.8571 | 1.0000 | 1.0000 | 0.9190 | 1.0000 | 1.0000 | 0.0000 | 0.127 | 0.127 | 0.144 |
| hybrid | 0.6857 | 0.3143 | 0.2000 | 0.6857 | 0.9429 | 1.0000 | 0.8129 | 1.0000 | 1.0000 | 0.0000 | 0.270 | 0.267 | 0.289 |

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
| semantic | ambiguous | 7 | 0.8571 | 0.4214 | 0.8571 |
| semantic | exact_keyword | 14 | 1.0000 | 0.8929 | 1.0000 |
| semantic | paraphrase | 14 | 0.7857 | 0.3833 | 0.7857 |
| keyword | ambiguous | 7 | 1.0000 | 0.8333 | 1.0000 |
| keyword | exact_keyword | 14 | 1.0000 | 1.0000 | 1.0000 |
| keyword | paraphrase | 14 | 1.0000 | 0.8810 | 1.0000 |
| hybrid | ambiguous | 7 | 1.0000 | 0.6548 | 1.0000 |
| hybrid | exact_keyword | 14 | 1.0000 | 0.9643 | 1.0000 |
| hybrid | paraphrase | 14 | 1.0000 | 0.7405 | 1.0000 |

## Performance by category

| Strategy | Category | Cases | Hit rate | MRR | Recall@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| semantic | athena_troubleshooting | 5 | 0.8000 | 0.6000 | 0.8000 |
| semantic | cost_awareness | 5 | 0.8000 | 0.4800 | 0.8000 |
| semantic | glue_troubleshooting | 5 | 0.8000 | 0.6000 | 0.8000 |
| semantic | iam_least_privilege | 5 | 1.0000 | 0.4667 | 1.0000 |
| semantic | monitoring | 5 | 1.0000 | 0.6000 | 1.0000 |
| semantic | pyspark_transformations | 5 | 1.0000 | 0.7500 | 1.0000 |
| semantic | s3_architecture | 5 | 0.8000 | 0.6667 | 0.8000 |
| keyword | athena_troubleshooting | 5 | 1.0000 | 0.8000 | 1.0000 |
| keyword | cost_awareness | 5 | 1.0000 | 0.8667 | 1.0000 |
| keyword | glue_troubleshooting | 5 | 1.0000 | 1.0000 | 1.0000 |
| keyword | iam_least_privilege | 5 | 1.0000 | 0.7667 | 1.0000 |
| keyword | monitoring | 5 | 1.0000 | 1.0000 | 1.0000 |
| keyword | pyspark_transformations | 5 | 1.0000 | 1.0000 | 1.0000 |
| keyword | s3_architecture | 5 | 1.0000 | 1.0000 | 1.0000 |
| hybrid | athena_troubleshooting | 5 | 1.0000 | 0.8000 | 1.0000 |
| hybrid | cost_awareness | 5 | 1.0000 | 0.6900 | 1.0000 |
| hybrid | glue_troubleshooting | 5 | 1.0000 | 0.8667 | 1.0000 |
| hybrid | iam_least_privilege | 5 | 1.0000 | 0.6667 | 1.0000 |
| hybrid | monitoring | 5 | 1.0000 | 0.8000 | 1.0000 |
| hybrid | pyspark_transformations | 5 | 1.0000 | 1.0000 | 1.0000 |
| hybrid | s3_architecture | 5 | 1.0000 | 0.8667 | 1.0000 |

## Performance by difficulty

| Strategy | Difficulty | Cases | Hit rate | MRR | Recall@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| semantic | easy | 14 | 1.0000 | 0.8929 | 1.0000 |
| semantic | hard | 10 | 0.7000 | 0.2667 | 0.7000 |
| semantic | medium | 11 | 0.9091 | 0.5136 | 0.9091 |
| keyword | easy | 14 | 1.0000 | 1.0000 | 1.0000 |
| keyword | hard | 10 | 1.0000 | 0.8333 | 1.0000 |
| keyword | medium | 11 | 1.0000 | 0.8939 | 1.0000 |
| hybrid | easy | 14 | 1.0000 | 0.9643 | 1.0000 |
| hybrid | hard | 10 | 1.0000 | 0.6367 | 1.0000 |
| hybrid | medium | 11 | 1.0000 | 0.7803 | 1.0000 |

## Failure summary

### semantic

- Missed expected targets at k=5: s3-ambiguous-landing, glue-paraphrase-input, athena-paraphrase-columnar, cost-paraphrase-safe-savings
- Returned no results: none

### keyword

- Missed expected targets at k=5: none
- Returned no results: none

### hybrid

- Missed expected targets at k=5: none
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
    "client_id": "demo-client",
    "environment": "dev"
  },
  "semantic_minimum_similarity": 0.0,
  "top_k": 5
}
```

Latency is local wall-clock time and is environment-dependent. Each reported query latency is the median of the configured repetitions. Ranking and quality metrics are deterministic for the versioned corpus, benchmark, and settings.
