# Local Ollama and Qdrant RAG

## Purpose

This optional path adds local retrieval-augmented generation (RAG). Ollama
generates embeddings and chat responses; Qdrant persists and searches chunk
vectors. It does not train, fine-tune, or otherwise modify a model.

The existing fake, Bedrock, in-memory, S3, CDK, evaluation, prompt, and safety
paths remain available:

```text
document -> existing validation/extraction/hash/chunking
         -> pending embedding artifact
         -> EmbeddingWorkflow -> OllamaEmbeddingProvider
         -> VectorIndexingWorkflow -> QdrantVectorStore
query    -> same embedding provider -> scoped Qdrant query
         -> existing GroundedPromptBuilder -> OllamaLLMProvider
```

Ollama and Qdrant are host-managed prerequisites. Installing Python
dependencies does not install either service or pull any model.

## Windows PowerShell setup

Install Ollama separately from its official distribution, then explicitly pull
the chosen models:

```powershell
ollama pull embeddinggemma
ollama pull qwen3:8b
ollama list
```

Start only Qdrant through Compose:

```powershell
docker compose up -d qdrant
docker compose ps qdrant
```

The official `qdrant/qdrant:v1.18.3-unprivileged` image stores data in the
named `qdrant_storage` volume. HTTP port 6333 and gRPC port 6334 bind to
`127.0.0.1`, not all host interfaces. The Streamlit app and Ollama remain on
the host.

Configure and start Streamlit:

```powershell
$env:APP_RUNTIME_MODE = "local"
$env:LLM_PROVIDER = "ollama"
$env:EMBEDDING_PROVIDER = "ollama"
$env:VECTOR_STORE_PROVIDER = "qdrant"
$env:OLLAMA_URL = "http://localhost:11434"
$env:OLLAMA_EMBEDDING_MODEL = "embeddinggemma"
$env:OLLAMA_CHAT_MODEL = "qwen3:8b"
$env:QDRANT_URL = "http://localhost:6333"
$env:QDRANT_COLLECTION = "dea_knowledge_embeddinggemma_v1"
$env:QDRANT_API_KEY = ""
python -m streamlit run ui/app.py
```

The equivalent `APP_`-prefixed names in `.env.example` take precedence. The
sidebar displays selected providers, model names, and collection name. It does
not display the API key. A connection status appears only after **Test local
connections** is clicked; the check does not run inference.

## Ingest a document

The explicit local indexing command uses the existing ingestion pipeline and
then consumes its pending embedding descriptor. Supported types and size limits
remain those in `KnowledgeConfig`.

```powershell
python -m scripts.index_local_document .\path\runbook.md `
  --client-id demo-client `
  --environment dev `
  --namespace data-engineering `
  --domain general
```

Artifacts are preserved under `.local/knowledge-store/` using the same
`knowledge/raw`, `processed`, `chunks`, `embeddings`, and `metadata` keys used
by S3. This directory is ignored by Git. The document ID is derived from the
file SHA-256, embedding records are keyed by chunk, and Qdrant point UUIDs are
derived from normalized client/environment, document ID, chunk ID, chunk
checksum, and embedding model. Repeating the command therefore skips unchanged
embedding calls and updates the same Qdrant points.

The S3 event processor now supports an injected automatic indexing service and
reuses its existing retry/DLQ path for incomplete documents. The synthesized
stack keeps that service disabled and does not gain Ollama, Qdrant, Bedrock, or
public-network permissions. See
[Automatic vector indexing](AUTOMATIC_VECTOR_INDEXING.md).

## Query the knowledge base

In Streamlit, select the same client and environment used during indexing and
submit a knowledge question. `RAGApplicationService` embeds the query with the
configured embedding provider, requires client/environment filters in the
Qdrant request, applies optional `knowledge_namespace`, `knowledge_domain`,
`agent`, `document_type`, and `source` filters, and passes only adequate results
through the existing evidence-first prompt builder. Empty retrieval continues
to return the existing insufficient-evidence response.

The default local bootstrap also indexes the repository's synthetic demo corpus
when local mode is explicitly activated. This is a bounded Ollama/Qdrant action,
not an import-time or ordinary demo-mode action.

## Client isolation and payload safety

One shared collection is used so deployments do not need a collection per
client. Every point carries normalized `client_id` and `environment` payloads.
Every `QdrantVectorStore.retrieve` call requires both fields and constructs
Qdrant `must` conditions before issuing `query_points`; callers cannot override
the scope fields with optional filters. Application-level scope checks remain
as defense in depth.

Payloads retain chunk text, source, document/chunk identity, model, timestamp,
namespace, and sanitized original metadata. Secret-like metadata keys, SSN-like
values, and long account/card-number-like digit sequences are excluded or
redacted. This is not a substitute for upstream data classification or a
reviewed DLP system. Do not index passwords, tokens, full account numbers, card
credentials, Social Security numbers, or unapproved client data.

Retrieved text remains untrusted reference material. Existing prompt-injection,
approval, insufficient-evidence, and non-destructive bookkeeping controls are
unchanged. Retrieval never authorizes a bookkeeping write or AWS action.

## Provider selection

These combinations are supported without code changes:

| LLM | Embeddings | Vector store | Use |
| --- | --- | --- | --- |
| `fake` | `fake` | `memory` | deterministic offline demo/tests |
| `bedrock` | `bedrock` | `memory` | existing AWS-backed mode |
| `ollama` | `ollama` | `qdrant` | persistent local RAG |

Provider variables are independently parsed, so deterministic fake embeddings
can also evaluate Qdrant without Ollama. Unsupported provider names and missing
selected local settings fail during configuration. Unselected providers are not
constructed or contacted.

Switch back to the fully offline default:

```powershell
$env:APP_RUNTIME_MODE = "demo"
$env:LLM_PROVIDER = "fake"
$env:EMBEDDING_PROVIDER = "fake"
$env:VECTOR_STORE_PROVIDER = "memory"
python -m streamlit run ui/app.py
```

## Model and collection changes

Do not mix incompatible embedding models in one collection. Choose a new,
normalized collection name when changing an embedding model, for example:

```powershell
$env:OLLAMA_EMBEDDING_MODEL = "another-reviewed-model"
$env:QDRANT_COLLECTION = "dea_knowledge_another_reviewed_model_v1"
```

Qdrant dimensions are discovered from the first embedding upsert. An existing
collection is reused only when its size and cosine distance match. The adapter
never silently deletes or recreates a collection. A model change requires a new
collection or a deliberate complete re-embedding into an empty compatible
collection.

## Reset only local Qdrant data

This destroys the local Qdrant index. First stop Qdrant and inspect the exact
named volume:

```powershell
docker compose stop qdrant
docker volume ls --filter name=qdrant_storage
```

After confirming the exact project-scoped volume name, remove that one volume
(the default Compose project name normally produces the following name), then
restart Qdrant:

```powershell
docker volume rm data-engineering-assistant-cdk_qdrant_storage
docker compose up -d qdrant
```

Do not use a wildcard or remove unrelated volumes. `.local/knowledge-store`
artifacts are separate and remain available for deliberate re-indexing.

## Troubleshooting and limitations

- Ollama unavailable: run `ollama list`, confirm port 11434, and confirm the
  configured model was explicitly pulled.
- Qdrant unavailable: run `docker compose ps qdrant` and
  `docker compose logs qdrant`; confirm port 6333 is not occupied.
- Dimension mismatch: select a new collection name or deliberately reset and
  fully re-embed the collection.
- No results: confirm client, environment, namespace, collection, model, top-k,
  and score threshold match indexing.
- Windows storage: the named Docker volume avoids known bind-mount filesystem
  issues for Qdrant data.
- The Compose service is a single local node without TLS or authentication. It
  is suitable for local development because it binds only to loopback. Never
  publish an unauthenticated Qdrant port; remote use requires HTTPS, an API key,
  backups, monitoring, and an availability design.
- Local inference latency and quality depend on model, CPU/GPU, memory, and
  corpus. Electricity, hardware, hosting, and operations costs are not included
  in the application cost estimate. Missing usage is never fabricated.

Official references:

- [Ollama download](https://ollama.com/download)
- [Ollama embed API](https://docs.ollama.com/api/embed)
- [Qdrant local quickstart](https://qdrant.tech/documentation/quick-start/)
- [Qdrant self-hosted security](https://qdrant.tech/documentation/security/)
