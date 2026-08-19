"""Chunking interfaces and the baseline text chunker."""

from typing import Protocol, runtime_checkable

from knowledge.models import KnowledgeChunk


@runtime_checkable
class Chunker(Protocol):
    """Interface for document-aware text chunking strategies."""

    def chunk(self, document_id: str, text: str) -> list[KnowledgeChunk]:
        """Split text while retaining references to the source document."""


class TextChunker:
    """Character-based chunker with deterministic overlap."""

    def __init__(self, chunk_size: int, overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "overlap must be non-negative and smaller than chunk_size"
            )
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, document_id: str, text: str) -> list[KnowledgeChunk]:
        if not document_id:
            raise ValueError("document_id cannot be empty")
        if not text:
            return []

        chunks: list[KnowledgeChunk] = []
        start = 0
        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            index = len(chunks)
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{document_id}:{index:06d}",
                    document_id=document_id,
                    index=index,
                    text=text[start:end],
                    start_character=start,
                    end_character=end,
                )
            )
            if end == len(text):
                break
            start = end - self._overlap

        return chunks
