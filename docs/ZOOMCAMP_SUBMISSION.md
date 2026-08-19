# LLM Zoomcamp submission map

This is a reviewer-facing evidence map and conservative score estimate. The
course platform owns the final peer-review score.

Official references, checked on 2026-08-18:

- [LLM Zoomcamp course documentation](https://datatalks.club/docs/courses/llm-zoomcamp/)
- [Current LLM Zoomcamp project guidance](https://datatalks.club/docs/courses/llm-zoomcamp/project/)
- [Course source repository](https://github.com/DataTalksClub/llm-zoomcamp)
- [Course management platform guidance](https://datatalks.club/docs/courses/course-management-platform/)

## Conservative score against the official rubric

The official project rubric awards 20 core points, three named best-practice
points, and up to five bonus points. This repository currently supports
**17/23 non-bonus points**, or **17/28 when the unclaimed bonus denominator is
included**.

| Official criterion | Points | Evidence and scoring rationale |
| --- | ---: | --- |
| Problem description | 2/2 | The README identifies the data-engineering audience, problem, public synthetic datasets, expected answers, and limitations. |
| Retrieval flow | 2/2 | Scoped knowledge retrieval, prompt construction, an LLM provider boundary, and cited responses form an end-to-end RAG flow. |
| Retrieval evaluation | 1/2 | Semantic, BM25, and RRF hybrid retrieval are compared on two corpora. BM25 wins, but the application intentionally retains its existing vector default. |
| LLM evaluation | 1/2 | Three prompt contracts are compared with reproducible fake providers. The winner is not promoted without representative real-model and independent review. |
| Interface | 2/2 | The Streamlit UI supports questions, cited answers, safety states, feedback, and offline monitoring. |
| Ingestion pipeline | 2/2 | The repository implements automated S3-event/Lambda ingestion plus local deterministic corpus loading. Deployment is optional for review. |
| Monitoring | 2/2 | The UI collects per-response feedback and presents a dashboard backed by six synthetic monitoring charts. |
| Containerization | 2/2 | The offline application and optional dependency services are defined in Compose; the default service requires no external infrastructure. |
| Reproducibility | 2/2 | Public datasets, exact constraints, offline defaults, tests, Docker health checks, and startup instructions are versioned together. |
| Hybrid search | 1/1 | BM25 plus vector retrieval is evaluated through reciprocal-rank fusion. |
| Document reranking | 0/1 | Not implemented or claimed. |
| Query rewriting | 0/1 | Not implemented or claimed. |
| Cloud deployment bonus | 0/2 | No deployment point is required or claimed for this submission. |
| Extra bonus | 0/3 | Left to peer-review discretion; no points are self-awarded. |

The two partial evaluation scores are deliberate and literal: the rubric's
two-point wording requires that the best measured approach be used. The
repository publishes the comparisons without misrepresenting deterministic
offline evidence as production-model quality.

| Area | Status | Repository evidence | Demonstrated level | Remaining work | Submission artifact |
| --- | --- | --- | --- | --- | --- |
| Problem description | Implemented | [README](../README.md), synthetic data-engineering scenarios | Strong project narrative and explicit users, scope, and safety constraints | Add a concise walkthrough showing the problem and outcome | README plus walkthrough |
| Retrieval flow | Implemented locally | [Embedding and retrieval](EMBEDDING_AND_RETRIEVAL.md), [retrieval comparison](RETRIEVAL_EVALUATION.md), `knowledge/retrieval.py`, `knowledge/keyword_retrieval.py`, `knowledge/hybrid_retrieval.py` | Scoped cosine, BM25, and reciprocal-rank fusion with deterministic ranking and result metadata | Evaluate a production-grade semantic model and larger corpus | Retrieval implementation and comparison report |
| Retrieval evaluation | Implemented for two synthetic corpora | [Retrieval comparison](RETRIEVAL_EVALUATION.md), [AWS operations dataset](AWS_PIPELINE_OPERATIONS_DATASET.md), committed JSON/Markdown/CSV results | Original 35-query benchmark plus a leakage-safe 36-query test split over the 36-document corpus; semantic/BM25/hybrid P@k, R@k, MRR, hit/no-result rates, grouping, latency, selection, and failures | Add independent judgments, hard negatives, and a production-grade semantic model comparison | Versioned benchmarks and reviewed result snapshots |
| LLM and prompt evaluation | Implemented for deterministic fake providers | [LLM and prompt evaluation](LLM_AND_PROMPT_EVALUATION.md), [AWS test-split summary](../evaluation/results/aws_pipeline_operations/evaluation_summary.md), committed JSON/Markdown/CSV results | Original 30 cases plus 18 test-split answer cases; three prompts, two fake modes, rule-based dimensions, grouped metrics, costs, selection, and failures | Run a representative real model and add an independent semantic or human review protocol | Versioned benchmark, comparison table, and explicit limitation |
| User interface | Implemented locally with reviewed captures | [Streamlit interface](STREAMLIT_INTERFACE.md), [reviewed screenshots](images/README.md), [private bookkeeping assistant](BOOKKEEPING_ASSISTANT.md), `ui/app.py` | Complete offline RAG demo plus bounded local bookkeeping analysis with deterministic metrics, explicit model approval, citations, safety statuses, feedback, and cost presentation | Record a concise walkthrough; Docker health capture remains separate | Screenshots and demo video |
| Knowledge ingestion | Implemented with an optional AWS trigger | [Knowledge layer](KNOWLEDGE_LAYER.md), [event-driven ingestion](EVENT_DRIVEN_INGESTION.md), `knowledge/ingestion.py`, `knowledge/event_ingestion.py`, `knowledge/pdf_extraction.py` | Metadata, checksums, chunks, manifest, S3 ObjectCreated handling, idempotency, scoped IAM, retries, pending embedding state, and offline text-based PDF extraction | Capture a sanitized deployed ingestion example; OCR and DOCX parsing remain deferred | Sanitized ingestion example |
| Monitoring and feedback | Implemented as offline synthetic evidence | [Monitoring and feedback analysis](MONITORING_AND_FEEDBACK.md), reviewed JSONL fixture, aggregate JSON/Markdown/CSV, six labeled PNG charts, and the Streamlit offline monitoring page | Typed scoped events, append-only local sink, 275 synthetic events, latency/cost/quality/safety/feedback metrics, privacy boundaries, and reviewer-facing evidence | Production persistence, alerting, retention, and dashboards require a separate reviewed design | Reviewed fixture, metrics, charts, and offline page |
| Containerization | Implemented | [Containerization](CONTAINERIZATION.md), `Dockerfile`, `compose.yaml` | Non-root Python 3.12 image, health check, offline default, Compose, Bedrock opt-in | Capture final build/health evidence; optionally add scanning or SBOM | Build log and health screenshot |
| Reproducibility | Implemented locally | `requirements.txt`, `requirements-dev.txt`, `constraints.txt`, scoped `pytest.ini`, CI workflow, deterministic providers | CI targets `main`, scopes tests to `tests/`, validates the dataset and its test split, synthesizes without lookups, and builds the constrained container | Repeat the full clean-clone/Docker/Streamlit verification and confirm hosted CI | Green GitHub Actions run |
| Engineering best practices | Implemented | Offline test suite, typed provider boundaries, configuration validation, structured logging, scoped retrieval, safety gates, documentation | Strong modularity, tests, security boundaries, and no secrets in configuration | Add public issue/PR templates only if project contribution volume warrants them | Test report and architecture docs |
| Cloud deployment bonus | Partial evidence | [Infrastructure](INFRASTRUCTURE.md), AWS CDK stack and synth tests | Reproducible AWS infrastructure definition with legacy stack compatibility | Provide sanitized deployment evidence or a reviewed hosted demo; no public service is currently claimed | Sanitized CloudFormation output or demo URL |

## Required evidence before submission

- [ ] Publish a sanitized, squashed snapshot to a dedicated public portfolio
  repository and verify it from a signed-out browser. Do not make the current
  private repository public because its earlier documentation history contains
  local workstation identity text that is absent from the reviewed snapshot.
- [ ] Run the complete release verification from a clean clone.
- [x] Publish retrieval comparisons for semantic, keyword, and hybrid methods.
- [x] Connect the 36-document dataset test split to retrieval and answer evaluation.
- [x] Publish an LLM/prompt comparison with selection rationale.
- [x] Add monitoring or feedback charts based only on synthetic or redacted data.
- [x] Add reviewed offline UI screenshots listed in [images/README.md](images/README.md).
- [ ] Add the Docker health capture after Docker Desktop is running.
- [ ] Record a concise setup and application walkthrough.
- [ ] Link a successful hosted CI run.
- [x] Recheck the current Zoomcamp rubric and use its exact scoring terminology.

## Submission form fields

- **Project title:** AWS Data Engineering Assistant: a scoped RAG assistant for
  pipeline design and troubleshooting
- **Repository URL:** create a dedicated public portfolio repository from the
  validated squashed snapshot, verify it from a signed-out browser, and paste
  its HTTPS URL
- **Commit hash:** create the submission commit after validation and paste the
  full 40-character hash from `git rev-parse HEAD`
- **Immutable review URL:** append `/tree/<full-commit-hash>` to the public
  repository URL

Do not submit a branch-only URL. Reviewers must be able to open the repository
and immutable commit without authentication.

The project should not add a managed vector store, public hosting,
authentication, or new AWS resources solely to imply completeness. Any such
change needs an explicit design, security, cost, and infrastructure review.
