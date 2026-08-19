# Offline LLM and prompt comparison

This report evaluates deterministic fake-provider adherence to prompt contracts. It does not measure real LLM quality and incurred no provider charge.

## Provenance

- Evaluation date: `2026-08-06T00:00:00Z`
- Git commit at generation: `0fd56f4faeb3f222dee66a5b1df128ad5f463bc2`
- Benchmark: `aws-pipeline-operations-answer-test-v1` (18 cases)
- Corpus: `aws-pipeline-operations-1.0.0`
- Retrieval: `fixed benchmark document IDs; no live retrieval`
- Fake modes: `concise, detailed`
- Prompt strategies: `baseline-concise` (`baseline-concise-v1`), `grounded-evidence-first` (`grounded-evidence-first-v1`), `structured-troubleshooting` (`structured-troubleshooting-v1`)
- Python: `3.12.13`
- Pricing profile: `prompt-eval-simulated-pricing-v1`
- Cost label: **simulated comparison estimate; no Bedrock charge**

## Strategy metrics

| Strategy | Grounded | Citation correct | Citation complete | Unsupported | Uncertainty | Safety | Approval | Complete | Format | Avg input | Avg output | Avg total | Simulated USD | Avg ms | P50 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline-concise | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.3333 | 1.0000 | 1250.1 | 30.8 | 1280.9 | 0.0003509652777777777777777777778 | 0.179 | 0.173 | 0.222 |
| grounded-evidence-first | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1272.9 | 57.0 | 1329.9 | 0.0003894861111111111111111111111 | 0.180 | 0.178 | 0.194 |
| structured-troubleshooting | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8333 | 1.0000 | 1269.9 | 60.2 | 1330.2 | 0.0003927638888888888888888888889 | 0.185 | 0.178 | 0.238 |

## Selection

- Recommended overall: **grounded-evidence-first**
- Best troubleshooting: **grounded-evidence-first**
- Best concise mode: **grounded-evidence-first**
- Best safety-sensitive: **grounded-evidence-first**
- Rule: Highest deterministic overall quality score, then answer completeness, then fewer total tokens.
- Versus baseline: 49.1 more average total tokens and USD 0.0000385208333333333333333333333 more simulated average cost.

The existing application prompt default was not changed.

## Performance by fake LLM mode

| Strategy | Mode | Quality | Grounded | Complete | Avg output tokens | Avg latency ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline-concise | concise | 0.9167 | 1.0000 | 0.0000 | 17.2 | 0.176 |
| baseline-concise | detailed | 0.9722 | 1.0000 | 0.6667 | 44.3 | 0.183 |
| grounded-evidence-first | concise | 0.9815 | 1.0000 | 1.0000 | 51.0 | 0.180 |
| grounded-evidence-first | detailed | 1.0000 | 1.0000 | 1.0000 | 63.0 | 0.180 |
| structured-troubleshooting | concise | 0.9259 | 1.0000 | 0.6667 | 49.9 | 0.183 |
| structured-troubleshooting | detailed | 1.0000 | 1.0000 | 1.0000 | 70.6 | 0.186 |

## Performance by category

| Strategy | Category | Quality | Grounded | Complete | Safety | Avg output | Avg ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline-concise | cost_optimization | 0.9583 | 1.0000 | 0.5000 | 1.0000 | 35.0 | 0.180 |
| baseline-concise | data_quality | 0.9583 | 1.0000 | 0.5000 | 1.0000 | 36.0 | 0.177 |
| baseline-concise | operational_procedure | 0.9167 | 1.0000 | 0.0000 | 1.0000 | 14.0 | 0.174 |
| baseline-concise | reliability | 0.9583 | 1.0000 | 0.5000 | 1.0000 | 44.0 | 0.175 |
| baseline-concise | schema_evolution | 0.9583 | 1.0000 | 0.5000 | 1.0000 | 44.5 | 0.174 |
| baseline-concise | security | 0.9583 | 1.0000 | 0.5000 | 1.0000 | 40.0 | 0.179 |
| baseline-concise | troubleshooting | 0.9583 | 1.0000 | 0.5000 | 1.0000 | 37.6 | 0.188 |
| baseline-concise | vector_operations | 0.9583 | 1.0000 | 0.5000 | 1.0000 | 44.5 | 0.172 |
| grounded-evidence-first | cost_optimization | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 54.0 | 0.191 |
| grounded-evidence-first | data_quality | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 55.0 | 0.173 |
| grounded-evidence-first | operational_procedure | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 52.0 | 0.180 |
| grounded-evidence-first | reliability | 0.9583 | 1.0000 | 1.0000 | 1.0000 | 64.0 | 0.175 |
| grounded-evidence-first | schema_evolution | 0.9583 | 1.0000 | 1.0000 | 1.0000 | 66.0 | 0.175 |
| grounded-evidence-first | security | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 58.0 | 0.189 |
| grounded-evidence-first | troubleshooting | 0.9931 | 1.0000 | 1.0000 | 1.0000 | 59.2 | 0.180 |
| grounded-evidence-first | vector_operations | 0.9583 | 1.0000 | 1.0000 | 1.0000 | 62.0 | 0.178 |
| structured-troubleshooting | cost_optimization | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 61.0 | 0.184 |
| structured-troubleshooting | data_quality | 0.9583 | 1.0000 | 1.0000 | 1.0000 | 62.0 | 0.178 |
| structured-troubleshooting | operational_procedure | 0.9583 | 1.0000 | 0.5000 | 1.0000 | 46.0 | 0.178 |
| structured-troubleshooting | reliability | 0.9583 | 1.0000 | 1.0000 | 1.0000 | 71.0 | 0.176 |
| structured-troubleshooting | schema_evolution | 0.9583 | 1.0000 | 1.0000 | 1.0000 | 73.0 | 0.246 |
| structured-troubleshooting | security | 0.9583 | 1.0000 | 1.0000 | 1.0000 | 70.0 | 0.193 |
| structured-troubleshooting | troubleshooting | 0.9653 | 1.0000 | 1.0000 | 1.0000 | 66.2 | 0.185 |
| structured-troubleshooting | vector_operations | 0.9583 | 1.0000 | 1.0000 | 1.0000 | 74.0 | 0.178 |

## Performance by difficulty

| Strategy | Difficulty | Quality | Grounded | Complete | Avg output | Avg ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline-concise | advanced | 0.9464 | 1.0000 | 0.3571 | 32.2 | 0.179 |
| baseline-concise | intermediate | 0.9375 | 1.0000 | 0.2500 | 25.8 | 0.181 |
| grounded-evidence-first | advanced | 0.9881 | 1.0000 | 1.0000 | 58.2 | 0.178 |
| grounded-evidence-first | intermediate | 1.0000 | 1.0000 | 1.0000 | 52.8 | 0.188 |
| structured-troubleshooting | advanced | 0.9613 | 1.0000 | 0.8571 | 61.7 | 0.185 |
| structured-troubleshooting | intermediate | 0.9688 | 1.0000 | 0.7500 | 55.1 | 0.186 |

## Performance by safety sensitivity

| Strategy | Group | Quality | Grounded | Complete | Safety | Avg output | Avg ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline-concise | safety_sensitive | 0.9583 | 1.0000 | 0.5000 | 1.0000 | 42.2 | 0.176 |
| baseline-concise | ordinary | 0.9427 | 1.0000 | 0.3125 | 1.0000 | 29.3 | 0.180 |
| grounded-evidence-first | safety_sensitive | 0.9792 | 1.0000 | 1.0000 | 1.0000 | 60.0 | 0.183 |
| grounded-evidence-first | ordinary | 0.9922 | 1.0000 | 1.0000 | 1.0000 | 56.6 | 0.180 |
| structured-troubleshooting | safety_sensitive | 0.9583 | 1.0000 | 1.0000 | 1.0000 | 72.0 | 0.186 |
| structured-troubleshooting | ordinary | 0.9635 | 1.0000 | 0.8125 | 1.0000 | 58.8 | 0.185 |

## Failure analysis

### baseline-concise

- unsupported answers: none
- missing citations: none
- unnecessary citations: none
- overlong responses: none
- incomplete troubleshooting steps: ae-apo-008-athena-query-cost-guardrails-failure:concise, ae-apo-009-athena-result-location-security-failure:concise, ae-apo-011-lambda-concurrency-backpressure-failure:concise, ae-apo-024-schema-registry-compatibility-failure:concise, ae-apo-033-embedding-validation-gate-failure:concise, ae-apo-034-qdrant-collection-safety-failure:concise
- safety failures: none
- approval gate failures: none
- uncertainty failures: none
- formatting failures: none

### grounded-evidence-first

- unsupported answers: none
- missing citations: none
- unnecessary citations: none
- overlong responses: ae-apo-011-lambda-concurrency-backpressure-control:concise, ae-apo-024-schema-registry-compatibility-control:concise, ae-apo-033-embedding-validation-gate-failure:concise, ae-apo-034-qdrant-collection-safety-control:concise
- incomplete troubleshooting steps: none
- safety failures: none
- approval gate failures: none
- uncertainty failures: none
- formatting failures: none

### structured-troubleshooting

- unsupported answers: none
- missing citations: none
- unnecessary citations: none
- overlong responses: ae-apo-008-athena-query-cost-guardrails-failure:concise, ae-apo-009-athena-result-location-security-control:concise, ae-apo-009-athena-result-location-security-failure:concise, ae-apo-011-lambda-concurrency-backpressure-control:concise, ae-apo-024-schema-registry-compatibility-control:concise, ae-apo-024-schema-registry-compatibility-failure:concise, ae-apo-033-embedding-validation-gate-control:concise, ae-apo-033-embedding-validation-gate-failure:concise, ae-apo-034-qdrant-collection-safety-control:concise, ae-apo-034-qdrant-collection-safety-failure:concise
- incomplete troubleshooting steps: none
- safety failures: none
- approval gate failures: none
- uncertainty failures: none
- formatting failures: none

## Interpretation

Scores are exact string, citation, flag, section, and token-limit checks against synthetic labels. They can verify deterministic orchestration, formatting, grounding markers, uncertainty, and safety gates. They cannot establish factual fluency, nuanced reasoning, naturalness, robustness to unseen prompts, or real model quality.
