"""Build selected fake, Bedrock, or local dependencies for the UI."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Protocol, Sequence

from knowledge.application import RAGApplicationService
from knowledge.bedrock_embeddings import BedrockEmbeddingProvider
from knowledge.bedrock_llm import BedrockLLMProvider
from knowledge.chunking import TextChunker
from knowledge.classification import RuleBasedIntentClassifier
from knowledge.config import (
    ApplicationConfig,
    ClassificationRoutingConfig,
)
from knowledge.costs import load_cost_estimator
from knowledge.fake_llm import DeterministicFakeLLMProvider
from knowledge.ollama_embeddings import OllamaEmbeddingProvider
from knowledge.ollama_llm import OllamaLLMProvider
from knowledge.models import EmbeddingRecord, KnowledgeChunk
from knowledge.prompting import GroundedPromptBuilder
from knowledge.retrieval import RetrievalEntry
from knowledge.routing import RequestRouter
from knowledge.qdrant_vector_store import QdrantVectorStore
from knowledge.vector_store import InMemoryVectorStore, VectorStore
from ui.config import (
    EmbeddingProviderName,
    LLMProviderName,
    RuntimeMode,
    UIConfig,
    VALID_UI_ENVIRONMENTS,
    VectorStoreProviderName,
)


DEMO_CORPUS_DIRECTORY = Path(__file__).with_name("demo_corpus")
DEMO_LICENSE = "CC0-1.0"
DEMO_EMBEDDING_MODEL_ID = "deterministic-demo-keyword-v1"
DEMO_CREATION_TIMESTAMP = datetime(
    2026,
    7,
    27,
    tzinfo=timezone.utc,
).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class DemoDocument:
    """One reproducible synthetic local knowledge document."""

    document_id: str
    title: str
    topic: str
    object_key: str
    text: str
    license: str = DEMO_LICENSE


@dataclass(frozen=True)
class RuntimeBundle:
    """Application plus safe runtime facts displayed by the UI."""

    application: RAGApplicationService
    runtime_mode: RuntimeMode
    corpus_document_count: int
    corpus_chunk_count: int
    embedding_provider_name: str
    llm_provider_name: str
    vector_store_provider_name: str = "memory"
    embedding_model_id: str | None = None
    llm_model_id: str | None = None
    vector_collection: str | None = None


class _EmbeddingProviderWithModel(Protocol):
    provider_name: str
    model_id: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input."""


class DeterministicDemoEmbeddingProvider:
    """Small keyword-hash embedder for reproducible offline demonstrations."""

    provider_name = "deterministic-demo"

    def __init__(
        self,
        *,
        model_id: str = DEMO_EMBEDDING_MODEL_ID,
        dimensions: int = 64,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id cannot be empty")
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.model_id = model_id.strip()
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding text cannot be empty")
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9]+", text.casefold())
        for token in tokens:
            digest = hashlib.sha256(
                f"{self.model_id}:{token}".encode("utf-8")
            ).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        magnitude = math.sqrt(sum(value * value for value in vector))
        if math.isclose(magnitude, 0.0):
            raise ValueError("Embedding text produced an empty vector")
        return [value / magnitude for value in vector]


def load_demo_documents(
    corpus_directory: Path | None = None,
) -> tuple[DemoDocument, ...]:
    """Load the repository-owned, synthetic Markdown corpus."""

    directory = corpus_directory or DEMO_CORPUS_DIRECTORY
    if not directory.is_dir():
        raise ValueError(f"Demo corpus directory not found: {directory}")
    metadata_by_filename = _load_corpus_metadata(directory)
    documents: list[DemoDocument] = []
    for path in sorted(directory.glob("*.md")):
        if path.name.casefold() == "readme.md":
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Demo document is empty: {path.name}")
        metadata = metadata_by_filename.get(path.name, {})
        title = (
            metadata.get("title")
            or _markdown_title(text)
            or path.stem.replace("-", " ").title()
        )
        document_id = metadata.get("document_id") or f"demo-{path.stem}"
        topic = (
            metadata.get("service")
            or metadata.get("domain")
            or path.stem
        )
        license_name = metadata.get("license") or DEMO_LICENSE
        object_key = (
            f"dataset://aws-pipeline-operations/{path.name}"
            if metadata
            else f"demo://synthetic/{path.name}"
        )
        documents.append(
            DemoDocument(
                document_id=document_id,
                title=title,
                topic=topic,
                object_key=object_key,
                text=text,
                license=license_name,
            )
        )
    if not documents:
        raise ValueError("Demo corpus contains no Markdown documents")
    return tuple(documents)


def build_runtime(
    config: UIConfig,
    *,
    runtime_mode: RuntimeMode | str | None = None,
    client_id: str | None = None,
    environment: str | None = None,
    retrieval_top_k: int | None = None,
    minimum_similarity: float | None = None,
    corpus_directory: Path | None = None,
    bedrock_runtime_client=None,
    ollama_http_session=None,
    qdrant_client=None,
    qdrant_models_module=None,
) -> RuntimeBundle:
    """Compose the existing application service for one explicit mode."""

    mode = RuntimeMode(runtime_mode or config.runtime_mode)
    scope_client = (
        client_id.strip().lower()
        if client_id is not None
        else config.default_client_id
    )
    scope_environment = (
        environment.strip().lower()
        if environment is not None
        else config.default_environment
    )
    top_k = (
        retrieval_top_k
        if retrieval_top_k is not None
        else config.retrieval_top_k
    )
    threshold = (
        minimum_similarity
        if minimum_similarity is not None
        else config.minimum_similarity
    )
    _validate_runtime_scope(
        scope_client,
        scope_environment,
        top_k,
        threshold,
    )

    documents = load_demo_documents(corpus_directory)
    llm_selection, embedding_selection, vector_selection = (
        provider_selection_for_mode(config, mode)
    )

    if embedding_selection == EmbeddingProviderName.FAKE:
        embedding_provider: _EmbeddingProviderWithModel = (
            DeterministicDemoEmbeddingProvider()
        )
    elif embedding_selection == EmbeddingProviderName.BEDROCK:
        embedding_provider = BedrockEmbeddingProvider(
            model_id=config.embedding_model_id,
            region_name=config.aws_region,
            bedrock_runtime_client=bedrock_runtime_client,
        )
    elif embedding_selection == EmbeddingProviderName.OLLAMA:
        embedding_provider = OllamaEmbeddingProvider(
            base_url=config.ollama_url,
            model_id=config.ollama_embedding_model,
            http_session=ollama_http_session,
        )
    else:  # pragma: no cover - enum validation owns this path
        raise ValueError("Unsupported embedding provider configuration")

    if llm_selection == LLMProviderName.FAKE:
        llm_provider = DeterministicFakeLLMProvider(
            response_text=(
                "Demo mode produced a deterministic response. Use a listed "
                "example for a tailored demonstration."
            ),
            responses_by_query=_demo_responses(),
            model_id="deterministic-demo-llm-v1",
            provider_metadata={
                "mode": "offline",
                "corpus": "synthetic-cc0",
            },
        )
    elif llm_selection == LLMProviderName.BEDROCK:
        llm_provider = BedrockLLMProvider(
            model_id=config.llm_model_id,
            region_name=config.aws_region,
            bedrock_runtime_client=bedrock_runtime_client,
        )
    elif llm_selection == LLMProviderName.OLLAMA:
        llm_provider = OllamaLLMProvider(
            base_url=config.ollama_url,
            model_id=config.ollama_chat_model,
            http_session=ollama_http_session,
        )
    else:  # pragma: no cover - enum validation owns this path
        raise ValueError("Unsupported LLM provider configuration")

    entries, chunk_count = _build_entries(
        documents,
        embedding_provider,
        client_id=scope_client,
        environment=scope_environment,
    )
    vector_store: VectorStore
    if vector_selection == VectorStoreProviderName.MEMORY:
        vector_store = InMemoryVectorStore(
            entries,
            top_k=top_k,
            minimum_similarity=float(threshold),
        )
    elif vector_selection == VectorStoreProviderName.QDRANT:
        vector_store = QdrantVectorStore(
            url=config.qdrant_url,
            collection_name=config.qdrant_collection,
            api_key=config.qdrant_api_key,
            client=qdrant_client,
            models_module=qdrant_models_module,
        )
        vector_store.upsert(
            entries,
            client_id=scope_client,
            environment=scope_environment,
        )
    else:  # pragma: no cover - enum validation owns this path
        raise ValueError("Unsupported vector-store provider configuration")
    application_config = ApplicationConfig(
        bedrock_llm_region=config.aws_region,
        bedrock_llm_model_id=config.llm_model_id,
        maximum_conversation_messages=(
            config.maximum_conversation_messages
        ),
        maximum_retrieved_chunks=top_k,
        minimum_similarity=float(threshold),
    )
    routing_config = ClassificationRoutingConfig(
        default_retrieval_top_k=top_k,
    )
    application = RAGApplicationService(
        classifier=RuleBasedIntentClassifier(routing_config),
        router=RequestRouter(routing_config),
        embedding_provider=embedding_provider,
        retriever=None,
        vector_store=vector_store,
        prompt_builder=GroundedPromptBuilder(
            prompt_version=application_config.prompt_version
        ),
        llm_provider=llm_provider,
        config=application_config,
        cost_estimator=load_cost_estimator(
            config.pricing_catalog_path
        ),
        runtime_mode=(
            RuntimeMode.DEMO.value
            if llm_selection == LLMProviderName.FAKE
            else llm_selection.value
        ),
    )
    return RuntimeBundle(
        application=application,
        runtime_mode=mode,
        corpus_document_count=len(documents),
        corpus_chunk_count=chunk_count,
        embedding_provider_name=embedding_provider.provider_name,
        llm_provider_name=llm_provider.provider_name,
        vector_store_provider_name=vector_store.provider_name,
        embedding_model_id=embedding_provider.model_id,
        llm_model_id=getattr(llm_provider, "model_id", None),
        vector_collection=getattr(vector_store, "collection_name", None),
    )


def provider_selection_for_mode(
    config: UIConfig,
    runtime_mode: RuntimeMode | str,
) -> tuple[
    LLMProviderName,
    EmbeddingProviderName,
    VectorStoreProviderName,
]:
    """Resolve configured providers or an explicitly selected UI profile."""

    mode = RuntimeMode(runtime_mode)
    if mode == config.runtime_mode:
        return (
            config.llm_provider,
            config.embedding_provider,
            config.vector_store_provider,
        )
    if mode == RuntimeMode.DEMO:
        return (
            LLMProviderName.FAKE,
            EmbeddingProviderName.FAKE,
            VectorStoreProviderName.MEMORY,
        )
    if mode == RuntimeMode.BEDROCK:
        return (
            LLMProviderName.BEDROCK,
            EmbeddingProviderName.BEDROCK,
            VectorStoreProviderName.MEMORY,
        )
    return (
        LLMProviderName.OLLAMA,
        EmbeddingProviderName.OLLAMA,
        VectorStoreProviderName.QDRANT,
    )


def check_local_connections(
    config: UIConfig,
    *,
    runtime_mode: RuntimeMode | str,
    ollama_http_session=None,
    qdrant_client=None,
    qdrant_models_module=None,
) -> dict[str, str]:
    """Check only explicitly selected local dependencies, without inference."""

    llm_selection, embedding_selection, vector_selection = (
        provider_selection_for_mode(config, runtime_mode)
    )
    statuses: dict[str, str] = {}
    if (
        llm_selection == LLMProviderName.OLLAMA
        or embedding_selection == EmbeddingProviderName.OLLAMA
    ):
        provider = OllamaEmbeddingProvider(
            base_url=config.ollama_url,
            model_id=config.ollama_embedding_model,
            http_session=ollama_http_session,
        )
        provider.check_connection()
        statuses["Ollama"] = "connected"
    if vector_selection == VectorStoreProviderName.QDRANT:
        store = QdrantVectorStore(
            url=config.qdrant_url,
            collection_name=config.qdrant_collection,
            api_key=config.qdrant_api_key,
            client=qdrant_client,
            models_module=qdrant_models_module,
        )
        store.check_connection()
        statuses["Qdrant"] = "connected"
    if not statuses:
        raise ValueError("No local provider is selected")
    return statuses


def _build_entries(
    documents: Sequence[DemoDocument],
    embedding_provider: _EmbeddingProviderWithModel,
    *,
    client_id: str,
    environment: str,
) -> tuple[list[RetrievalEntry], int]:
    chunker = TextChunker(chunk_size=1_200, overlap=150)
    document_chunks: list[tuple[DemoDocument, KnowledgeChunk]] = []
    for document in documents:
        document_chunks.extend(
            (document, chunk)
            for chunk in chunker.chunk(document.document_id, document.text)
        )
    vectors = embedding_provider.embed(
        [chunk.text for _, chunk in document_chunks]
    )
    if len(vectors) != len(document_chunks):
        raise ValueError(
            "Embedding provider did not return one vector per demo chunk"
        )

    entries = []
    for (document, chunk), vector in zip(
        document_chunks,
        vectors,
        strict=True,
    ):
        checksum = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
        record = EmbeddingRecord(
            schema_version=EmbeddingRecord.CURRENT_SCHEMA_VERSION,
            document_id=document.document_id,
            chunk_id=chunk.chunk_id,
            chunk_text_checksum=checksum,
            embedding_model_id=embedding_provider.model_id,
            embedding_dimensions=len(vector),
            embedding_vector=tuple(vector),
            creation_timestamp=DEMO_CREATION_TIMESTAMP,
            source_object_key=document.object_key,
        )
        entries.append(
            RetrievalEntry(
                embedding_record=record,
                source=document.title,
                text=chunk.text,
                metadata={
                    "client_id": client_id,
                    "environment": environment,
                    "object_key": document.object_key,
                    "section": document.title,
                    "topic": document.topic,
                    "file_type": "markdown",
                    "object_classification": "indexable_text_document",
                    "indexable": True,
                    "storage_only": False,
                    "synthetic": True,
                    "license": document.license,
                },
            )
        )
    return entries, len(document_chunks)


def _validate_runtime_scope(
    client_id: str,
    environment: str,
    top_k: int,
    threshold: float,
) -> None:
    if not re.fullmatch(
        r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?",
        client_id,
    ):
        raise ValueError("client_id is invalid")
    if environment not in VALID_UI_ENVIRONMENTS:
        raise ValueError("environment is invalid")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 50:
        raise ValueError("retrieval_top_k must be between 1 and 50")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not -1.0 <= float(threshold) <= 1.0
    ):
        raise ValueError(
            "minimum_similarity must be between -1 and 1"
        )


def _markdown_title(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _load_corpus_metadata(directory: Path) -> dict[str, dict[str, str]]:
    metadata_path = directory.parent / "metadata" / "documents.jsonl"
    if not metadata_path.is_file():
        return {}
    records: dict[str, dict[str, str]] = {}
    for line_number, line in enumerate(
        metadata_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid corpus metadata at line {line_number}: "
                f"{metadata_path}"
            ) from error
        if not isinstance(raw, dict):
            raise ValueError(
                f"Corpus metadata line {line_number} must be an object"
            )
        filename = raw.get("filename")
        document_id = raw.get("document_id")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(
                f"Corpus metadata line {line_number} has no filename"
            )
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(
                f"Corpus metadata line {line_number} has no document_id"
            )
        if filename in records:
            raise ValueError(f"Duplicate corpus metadata filename: {filename}")
        records[filename] = {
            key: value.strip()
            for key, value in raw.items()
            if key in {"document_id", "title", "service", "domain", "license"}
            and isinstance(value, str)
            and value.strip()
        }
    return records


def _demo_responses() -> dict[str, str]:
    return {
        "Design an S3-to-Glue-to-Athena pipeline.": (
            "Use an encrypted, versioned S3 landing zone, a least-privilege "
            "Glue role, cataloged curated data, and an Athena workgroup with "
            "a controlled results location. These are recommendations; "
            "confirm volume, latency, schema, and retention requirements."
        ),
        "Why did my Glue job fail with an access-denied error?": (
            "The synthetic troubleshooting guide recommends identifying the "
            "denied API and ARN, then checking the Glue execution role, bucket "
            "policy, KMS permissions, and CloudWatch evidence [S1]."
        ),
        "Write a PySpark deduplication transformation.": (
            "```python\n"
            "from pyspark.sql import functions as F, Window\n\n"
            "window = Window.partitionBy(\"business_key\").orderBy(\n"
            "    F.col(\"updated_at\").desc()\n"
            ")\n"
            "result = (source.withColumn(\"_rank\", F.row_number().over(window))\n"
            "          .filter(F.col(\"_rank\") == 1).drop(\"_rank\"))\n"
            "```\n"
            "Assumption: `updated_at` is a reliable ordering field."
        ),
        "What information do you need before designing my pipeline?": (
            "Please provide source and target systems, formats, volume, "
            "arrival pattern, latency SLA, schema-change expectations, data "
            "quality rules, security classification, retention, recovery "
            "requirements, and cost constraints."
        ),
        "Deploy my CDK stack.": (
            "No deployment is performed in this interface."
        ),
        "Delete the production data bucket.": (
            "No destructive action is performed in this interface."
        ),
    }
