"""Document parsing and chunking utilities for the demo project."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    """A small searchable text unit."""

    chunk_id: str
    document_id: str
    content: str
    start: int
    end: int


def normalize_text(text: str) -> str:
    """Normalize whitespace so retrieval input is stable."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return " ".join(text.split())


def split_text_into_chunks(
    *,
    document_id: str,
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[TextChunk]:
    """Split one document into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    normalized = normalize_text(text)

    if not normalized:
        return []

    chunks: list[TextChunk] = []
    start = 0

    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        content = normalized[start:end]
        chunk_index = len(chunks)

        chunks.append(
            TextChunk(
                chunk_id=f"{document_id}::chunk::{chunk_index}",
                document_id=document_id,
                content=content,
                start=start,
                end=end,
            )
        )

        if end >= len(normalized):
            break

        start = end - overlap

    return chunks
