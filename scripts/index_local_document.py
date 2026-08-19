"""Explicitly ingest one local document into Ollama and Qdrant."""

import argparse
import hashlib
import json
from pathlib import Path

from knowledge.embedding_workflow import EmbeddingWorkflow
from knowledge.ingestion import KnowledgeIngestionPipeline
from knowledge.manifest import KnowledgeManifestRepository
from knowledge.ollama_embeddings import OllamaEmbeddingProvider
from knowledge.qdrant_vector_store import QdrantVectorStore
from knowledge.storage import FileSystemKnowledgeStorage
from knowledge.vector_indexing import VectorIndexingWorkflow
from ui.config import (
    EmbeddingProviderName,
    VectorStoreProviderName,
    load_ui_config,
)


def main() -> int:
    """Run the opt-in local indexing path without downloading models."""

    parser = argparse.ArgumentParser(
        description="Index one local document with Ollama and Qdrant."
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--client-id", required=True)
    parser.add_argument(
        "--environment",
        choices=("dev", "test", "stage", "prod"),
        default="dev",
    )
    parser.add_argument("--namespace", default="data-engineering")
    parser.add_argument("--domain", default="general")
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=Path(".local/knowledge-store"),
    )
    arguments = parser.parse_args()

    source_path = arguments.path.resolve()
    if not source_path.is_file():
        parser.error("path must identify a readable file")
    content = source_path.read_bytes()
    document_hash = hashlib.sha256(content).hexdigest()
    document_id = f"local-{document_hash[:24]}"

    config = load_ui_config()
    if config.embedding_provider != EmbeddingProviderName.OLLAMA:
        parser.error("EMBEDDING_PROVIDER must be ollama")
    if config.vector_store_provider != VectorStoreProviderName.QDRANT:
        parser.error("VECTOR_STORE_PROVIDER must be qdrant")

    storage = FileSystemKnowledgeStorage(arguments.storage_root)
    manifest = KnowledgeManifestRepository(storage)
    entry = KnowledgeIngestionPipeline(
        storage,
        manifest=manifest,
        document_id_factory=lambda: document_id,
    ).ingest(
        filename=source_path.name,
        content=content,
        source=f"local://{source_path.name}",
    )
    embedding_provider = OllamaEmbeddingProvider(
        base_url=config.ollama_url,
        model_id=config.ollama_embedding_model,
    )
    vector_store = QdrantVectorStore(
        url=config.qdrant_url,
        collection_name=config.qdrant_collection,
        api_key=config.qdrant_api_key,
    )
    report = VectorIndexingWorkflow(
        storage=storage,
        embedding_workflow=EmbeddingWorkflow(
            storage=storage,
            provider=embedding_provider,
            model_id=embedding_provider.model_id,
            batch_size=8,
            manifest=manifest,
        ),
        vector_store=vector_store,
        manifest=manifest,
    ).index_pending_document(
        entry,
        client_id=arguments.client_id,
        environment=arguments.environment,
        knowledge_namespace=arguments.namespace,
        knowledge_domain=arguments.domain,
    )
    print(
        json.dumps(
            {
                "document_id": report.document_id,
                "embedding_model": report.embedding_report.model_id,
                "embedding_created": len(report.embedding_report.created),
                "embedding_skipped": len(
                    report.embedding_report.skipped_chunk_ids
                ),
                "vector_status": report.vector_status.value,
                "vector_store": report.vector_store_provider,
                "vector_collection": report.vector_collection,
                "upserted_count": report.upserted_count,
                "already_indexed_count": (
                    report.statistics.already_indexed_chunk_count
                ),
                "indexed_chunk_count": report.statistics.indexed_chunk_count,
                "pending_chunk_count": report.statistics.pending_chunk_count,
                "failed_chunk_count": report.statistics.failed_chunk_count,
                "vector_dimension": report.vector_dimension,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
