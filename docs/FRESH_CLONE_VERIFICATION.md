# Fresh-clone release verification (historical baseline)

> Status note, 2026-08-06: this document records the committed 2026-07-27
> baseline and is not evidence that the current advanced working tree has been
> reproduced from a fresh clone. The current tree has since added production
> ingestion/indexing, Qdrant, media isolation, bookkeeping, and the AWS
> operations dataset. CI now targets `main`, scopes pytest to `tests/`, uses
> no-lookup synthesis, validates the dataset test split, and installs container
> dependencies with `constraints.txt`. A new full clean-clone, container health,
> Streamlit screenshot, and hosted-workflow run remain required before final
> submission.

## Conclusion

**PASS** for local Community Edition reproducibility on Windows. A clean clone
installed, compiled, passed all tests, regenerated the documented evaluation
evidence, launched the offline Streamlit application, built the Linux
container, validated the Compose configuration, and synthesized the
legacy-compatible CDK stack.

This verification did not deploy or destroy infrastructure, run `cdk diff`,
call an AWS API, invoke Amazon Bedrock, push an image, push a commit, or create
a tag.

## Verification environment

| Item | Verified value |
| --- | --- |
| Date | 2026-07-27 |
| Host | Microsoft Windows 11 Home, build 26200 |
| PowerShell | 5.1.26100.8875 |
| Git | 2.55.0.windows.3 |
| Python | 3.12.10 |
| pip | 26.1.2 |
| Node.js | 24.18.0 |
| AWS CDK CLI | 2.1133.0 |
| Docker Engine | 29.6.1 |
| Docker Compose | 5.3.0 |
| Container target | Linux/amd64 |

The final environment resolved, among other packages, pytest 8.4.2, Streamlit
1.60.0, boto3 1.43.57, and aws-cdk-lib 2.262.1 from the committed constraints
and requirements.

## Clean-clone method

The source worktree was confirmed clean with:

```powershell
git status
git log --oneline -10
git ls-files --others --exclude-standard
git diff --check
```

An isolated directory outside the source repository was created with:

```powershell
git clone --no-local <local-source-repository> <temporary-verification-directory>
Set-Location <temporary-verification-directory>
```

Because Git created the verification copy, it contained only committed files.
It did not copy a virtual environment, `cdk.out`, caches, local monitoring
events, `.env`, Streamlit secrets, AWS credentials, or generated response
files.

## Installation, compilation, and tests

The documented constrained installation was used:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --constraint constraints.txt --requirement requirements.txt --requirement requirements-dev.txt
python --version
python -m pip --version
python -m compileall knowledge ui evaluation tests
python -m pytest
git diff --check
```

All commands succeeded in the final clone. The result was **355 passed, 0
failed** in 17.33 seconds. Compilation completed without an error, and the
pre-evaluation whitespace check was clean.

## Evaluation evidence

The exact documented commands succeeded:

```powershell
python -m evaluation.run_retrieval_comparison
python -m evaluation.run_llm_prompt_comparison
python -m evaluation.generate_monitoring_fixture
python -m evaluation.run_monitoring_report
```

The retrieval comparison evaluated 35 cases. Keyword retrieval remained the
offline recommendation:

| Strategy | MRR | Hit@5 |
| --- | ---: | ---: |
| Semantic | 0.5948 | 0.8857 |
| Keyword | 0.9190 | 1.0000 |
| Hybrid | 0.8129 | 1.0000 |

The prompt comparison evaluated 30 cases, producing 180
case/strategy/mode results. Grounded evidence-first remained the recommended
strategy with 1.0000 overall quality and complete citations.

The monitoring generator verified the existing reviewed fixture without
rewriting it. The report analyzed 275 synthetic events across 84 requests:

| Metric | Result |
| --- | ---: |
| Request success | 0.9524 |
| Positive feedback | 0.7381 |
| Complete citations | 0.8750 |

All expected JSON, Markdown, and CSV files existed, and all six expected PNG
charts existed. The charts were byte-identical to the committed versions. The
275 JSONL fixture records were identical after platform line-ending
normalization.

Runtime evaluation timestamps and measured latencies changed as documented.
After excluding only those provenance/runtime fields, the retrieval, prompt,
and monitoring JSON outputs were identical to the reviewed results. Parsed CSV
records were also identical after excluding measured latency columns. Windows
line-ending conversion made some generated CSV files appear modified, but
introduced no record change. No metric, ranking, recommendation, checksum,
fixture record, or chart changed unexpectedly.

## Verified fresh-clone defects and corrections

The first isolated run exposed two genuine reproducibility defects:

1. `python -m evaluation.generate_monitoring_fixture` exited with status 2
   when the matching reviewed fixture already existed, even though the README
   prescribed that exact command.
2. The retrieval corpus checksum depended on checkout line endings, so the
   same committed corpus produced a different checksum on Windows.

The smallest corrections were applied:

- A matching existing monitoring fixture is now validated and reported as
  verified without being rewritten. A different existing fixture still fails,
  and `--force` remains required for intentional replacement.
- Retrieval corpus hashing normalizes CRLF and CR to LF before computing the
  checksum.
- Focused tests cover idempotent verification, altered-fixture rejection,
  forced replacement, and line-ending-independent corpus checksums.

The focused regression suite passed 67 tests before the complete fixed-clone
verification. No application behavior or CDK infrastructure was changed.

## Streamlit

The application was launched from the clone with:

```powershell
python -m streamlit run ui/app.py
```

The Streamlit health endpoint returned HTTP 200 with `ok`. The running process
was stopped after verification.

Streamlit's application test interface then exercised the default demo with
AWS client construction replaced by a sentinel that would fail the test. The
assistant page rendered a grounded answer, an `[S1]` citation, citation
details, cost information, and the offline-mode label. The monitoring page
rendered its heading, six metrics, synthetic-data warning, and download
control. There were no application-test errors and the AWS sentinel was never
called, proving that neither credentials nor a Bedrock request were needed.

## Docker

The required commands succeeded:

```powershell
docker build -t data-engineering-assistant:fresh-clone .
docker compose config
```

The build produced a Linux/amd64 image. A second exact build confirmed the
cached build path after the initial command wrapper timed out after Docker had
already printed successful image creation. Compose rendered a valid
offline-demo configuration, including its health check. The optional container
run was not needed because the host Streamlit process and health endpoint had
already been verified. No image was pushed.

## CDK

Only offline synthesis was run:

```powershell
cdk.cmd synth -c client=internal-dev
```

Synthesis succeeded and included the required legacy stack identity:

```text
DataEngineeringAssistantCdkStack
```

No `cdk diff`, deploy, destroy, or AWS API call was performed.

## Security and privacy review

A tracked-file name review and content-pattern scan found no committed `.env`,
Streamlit secrets file, `.aws` directory, private key, credential file,
purchased book, customer document, or private prompt collection. No GitHub,
OpenAI, Slack, bearer-token, or private-key pattern was found.

The only AWS-key-shaped and password/secret-assignment matches were deliberate
invalid sentinels in privacy unit tests and a clearly marked placeholder in
container documentation. Manual inspection confirmed that none was a usable
secret. No secret value was printed during verification.

The knowledge corpus, evaluation benchmarks, and monitoring fixture are
explicitly synthetic and CC0. Local monitoring output remains ignored by Git
and excluded from the container build context.

## Documentation review

The README documents Python 3.12 setup, constrained installation, pytest,
offline Streamlit startup, Docker, CDK synthesis, every evaluation command,
monitoring, architecture, limitations, licenses, example questions, screenshot
locations, and Windows PowerShell syntax. Local Markdown links resolve.
Infrastructure, knowledge, embedding/retrieval, classification/routing,
end-to-end RAG, Streamlit, containerization, cost estimation, monitoring, and
Zoomcamp evidence each have dedicated documentation.

Two pieces of release knowledge remain external or incomplete:

- The local repository has no configured Git remote, so an exact public
  `git clone` URL cannot be documented without inventing a destination. Add the
  final GitHub URL to the README when the repository is published.
- No real application screenshots are committed. The reviewed capture and
  redaction checklist is in `docs/images/README.md`.

The current official LLM Zoomcamp course and project guidance is linked from
the submission map. Exact cohort deadlines, the submission window, and the
platform rubric remain externally managed and must be rechecked immediately
before submission.

## GitHub Actions parity

The local run matched or exceeded `.github/workflows/ci.yml`: Python 3.12
constrained installation, compilation, pytest, retrieval evaluation, prompt
evaluation, monitoring evaluation, whitespace checking, internal-dev CDK
synthesis, and the Docker build all passed.

CI writes evaluation output to its temporary directory and supplies fixed
provenance values, so it will not dirty reviewed artifacts. The workflow pins
Node.js 22 and CDK 2.1133.0; local synthesis used the same CDK version. The
Docker build also installed and ran the Python 3.12 dependency set on Linux,
which provides additional Linux-runner compatibility evidence. No
Windows-only path or line-ending failure remains known.

Hosted GitHub Actions has not yet been observed because this local repository
has no configured remote. A green hosted run remains a public-release
requirement.

## Known release limitations

- Real screenshots and a concise setup/application walkthrough are not yet
  published.
- A hosted GitHub Actions run is not yet linked.
- The exact public repository clone URL is not yet available.
- Real-model and human evaluation remain intentionally deferred; current
  prompt evidence uses deterministic fake providers.
- The application has no authentication, persistent conversations, production
  monitoring backend, managed vector store, or hosted production service.
- Bedrock pricing must be revalidated before any optional Bedrock
  demonstration.

These limitations do not block the verified offline Community Edition workflow,
but screenshots, a public clone URL, a hosted green workflow, and a walkthrough
remain blockers for a polished public/Zoomcamp submission.
