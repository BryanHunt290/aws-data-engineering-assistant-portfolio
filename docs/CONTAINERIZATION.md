# Containerization

The complete local Streamlit application is packaged as a single Docker image.
The default container is an offline demonstration and does not require an AWS
account, credentials, S3, Bedrock, a database, or a managed vector store.

## Architecture

```text
python:3.12.10-slim
└── /opt/data-engineering-assistant
    ├── knowledge/          provider-neutral application backend
    ├── ui/                 Streamlit UI and synthetic CC0 corpus
    └── requirements.txt    bounded Python dependencies
```

The image runs Streamlit as UID/GID `10001:10001`. It exposes port `8501` and
checks `http://127.0.0.1:8501/_stcore/health`. The health check contacts only
the local Streamlit process; it does not classify a request, load embeddings,
invoke a model, or call AWS.

## Prerequisites

- Docker Desktop or a compatible Docker Engine with Compose v2
- Enough local resources to build a Python slim image
- Network access to Python package indexes during the image build

Python 3.12 is the supported application version. Direct local development may
use Python 3.12 without Docker.

## Build and run the offline demo

Run these commands from the repository root in Windows PowerShell:

```powershell
docker build -t data-engineering-assistant .
docker run --rm -p 8501:8501 data-engineering-assistant
```

Open `http://localhost:8501`.

The image defaults `APP_RUNTIME_MODE=demo`. Demo bootstrap uses the local
synthetic CC0 corpus, deterministic fake embeddings, deterministic fake
generation, and the in-memory retriever. The image also includes the reviewed
synthetic monitoring fixture and aggregate evidence used by the read-only
**Offline monitoring** page. Local `data/monitoring` files remain excluded.
The default demo makes no AWS call.

To run in the background and inspect health:

```powershell
docker run --rm -d --name dea-container-test -p 8501:8501 data-engineering-assistant
docker inspect --format="{{json .State.Health}}" dea-container-test
docker stop dea-container-test
```

## Docker Compose

The default Compose project starts only the offline `demo` service:

```powershell
docker compose up --build
```

The service uses `restart: unless-stopped`, maps `${APP_PORT:-8501}` to the
container, and duplicates the image health check. Validate resolved Compose
configuration with:

```powershell
docker compose config
```

The `bedrock` service is behind an explicit profile and is not started by the
default command.

## Optional Bedrock mode

Bedrock mode can incur AWS charges. Confirm the intended Region, model access,
quotas, and pricing before starting it. The local synthetic corpus is embedded
at runtime, so initial startup also invokes the configured embedding model.

Conceptually, the caller needs only `bedrock:InvokeModel` for the exact
embedding and generation models selected. This container does not require S3
permissions. Do not broaden an existing IAM role or policy merely to run the
demo; use a separately reviewed least-privilege identity.

### Standard environment credential chain

For temporary, externally managed environment credentials, explicitly forward
the standard boto3 variables without writing values to Compose:

```powershell
$env:AWS_ACCESS_KEY_ID = "<temporary-access-key>"
$env:AWS_SECRET_ACCESS_KEY = "<temporary-secret-key>"
$env:AWS_SESSION_TOKEN = "<temporary-session-token>"
$env:AWS_REGION = "us-west-2"

docker compose --profile bedrock run --rm --service-ports `
  -e AWS_ACCESS_KEY_ID `
  -e AWS_SECRET_ACCESS_KEY `
  -e AWS_SESSION_TOKEN `
  bedrock
```

Use short-lived credentials and clear them from the shell afterward. Do not put
credential values in `.env`, `compose.yaml`, image layers, or documentation.

### Read-only shared AWS configuration

A reviewed AWS CLI profile can be mounted read-only with `docker run`:

```powershell
$env:AWS_PROFILE = "dea-bedrock-demo"
$env:AWS_REGION = "us-west-2"

docker run --rm -p 8501:8501 `
  -e APP_RUNTIME_MODE=bedrock `
  -e AWS_PROFILE `
  -e AWS_REGION `
  --mount "type=bind,source=$HOME\.aws,target=/home/app/.aws,readonly" `
  data-engineering-assistant
```

The mount is explicit and is not present in the Dockerfile or default Compose
configuration. Verify that Docker Desktop is permitted to read the host path.

## Runtime configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_RUNTIME_MODE` | `demo` | Explicit `demo` or `bedrock` mode |
| `AWS_REGION` | `us-west-2` | Bedrock Region |
| `APP_EMBEDDING_MODEL_ID` | `amazon.titan-embed-text-v2:0` | Embedding model |
| `APP_LLM_MODEL_ID` | `anthropic.claude-3-haiku-20240307-v1:0` | Converse model |
| `APP_DEFAULT_CLIENT_ID` | `demo-client` | Initial client scope |
| `APP_DEFAULT_ENVIRONMENT` | `dev` | Initial environment scope |
| `APP_RETRIEVAL_TOP_K` | `5` | Maximum retrieved chunks |
| `APP_MINIMUM_SIMILARITY` | `0.0` | Similarity threshold |
| `APP_MAXIMUM_CONVERSATION_MESSAGES` | `10` | Session history bound |
| `APP_DEVELOPER_MODE` | `false` | Local stack-trace opt-in |
| `APP_PRICING_CATALOG_PATH` | unset | Optional reviewed local JSON price catalog |
| `APP_PORT` | `8501` | Host port for the Compose demo service |
| `BEDROCK_APP_PORT` | `8502` | Host port for the profiled Bedrock service |

The earlier `DEA_*` application variable names remain accepted for direct local
compatibility. Container-oriented `APP_*` names take precedence.

`.env.example` contains non-secret defaults for reference. Real `.env` files,
AWS directories, credential files, local Compose overrides, and private key
formats are ignored from the build context.

## Image security decisions

- The base is the official `python:3.12.10-slim` image.
- Runtime uses non-root UID/GID 10001.
- Only `requirements.txt`, `knowledge/`, and `ui/` are copied.
- Tests, Git data, CDK output, virtual environments, logs, local output,
  credentials, and environment files are excluded.
- The synthetic corpus remains included beneath `ui/demo_corpus/`.
- Python output is unbuffered and bytecode generation is disabled.
- pip caching and Streamlit telemetry are disabled.
- No credential, account ID, secret, or AWS resource is embedded.
- The application still has no execution tool. Containerization does not
  bypass approval or destructive-action safety routes.

## Troubleshooting

**Port 8501 is already in use**

```powershell
$env:APP_PORT = "8502"
docker compose up --build
```

**The container is unhealthy**

```powershell
docker inspect --format="{{json .State.Health}}" dea-container-test
docker logs dea-container-test
```

The normal UI hides stack traces. Container logs should still be treated as
sensitive operational data even though complete prompts and credentials are
not intentionally logged.

**Bedrock credentials are missing**

Use an approved short-lived environment credential or read-only profile mount.
Do not add credentials to the image or Compose file.

**Bedrock access is denied or a model is unavailable**

Confirm Region, model access, the two exact model IDs, and least-privilege
`bedrock:InvokeModel` authorization. Demo mode remains available while Bedrock
configuration is corrected.

**A package build fails**

Confirm Docker can reach the configured Python package index and rebuild
without a stale intermediate layer:

```powershell
docker build --pull --no-cache -t data-engineering-assistant .
```

## Zoomcamp alignment

Containerization contributes to:

- **Reproducibility:** a supported Python version, bounded dependencies, and
  versioned offline corpus are packaged together.
- **Application interface:** reviewers receive the existing complete Streamlit
  UI and backend orchestration.
- **Complete packaging:** one image contains the application layers needed for
  the local demo.
- **Reviewer onboarding:** the default build and run commands need no AWS
  account or configuration.
- **Containerization rubric:** the image includes non-root operation, health
  checks, ignore rules, environment configuration, and Compose startup.

This does not claim complete Zoomcamp submission readiness. Final screenshots,
walkthrough material, evaluation reporting, rubric review, and any hosted
deployment remain deferred.

## Deferred work

- Image publishing, signing, SBOM generation, and vulnerability scanning
- Hosted container service and TLS ingress
- Authentication and multi-user session isolation
- Persistent feedback or conversation storage
- Managed vector storage
- Autoscaling and production health/metrics integration
- Execution tools or deployment automation

Each hosted or persistent capability requires a separate reviewed
infrastructure proposal.
