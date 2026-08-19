# AWS pipeline operations evaluation summary

Offline, deterministic evidence for the leakage-safe dataset split.

## Coverage

- Dataset version / split: `1.0.0` / `test`
- Documents: `6`
- Retrieval queries: `36` (`30` scored, `6` unanswerable)
- Answer cases: `18`

## Selections

- Recommended retrieval strategy: `keyword`
- Recommended prompt strategy: `grounded-evidence-first`

## Retrieval metrics

| Strategy | MRR | Hit rate | Recall@5 | No-result rate |
| --- | ---: | ---: | ---: | ---: |
| semantic | 0.3372 | 0.5000 | 0.5000 | 0.0000 |
| keyword | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| hybrid | 0.7306 | 0.9333 | 0.9333 | 0.0000 |

## Limitations

- Unanswerable retrieval cases are preserved and counted but are not included in precision, recall, or MRR calculations.
- Answer evaluation uses a deterministic fake provider to measure prompt-contract adherence, not real language-model quality.
- All documents and cases are synthetic; no provider call or deployment occurs.
