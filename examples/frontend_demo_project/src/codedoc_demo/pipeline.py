"""Search pipeline orchestration for the demo project."""

from __future__ import annotations

from collections.abc import Iterable

from codedoc_demo.parser import TextChunk, split_text_into_chunks
from codedoc_demo.search import search_documents


def build_chunks_from_documents(
    documents: Iterable[dict],
    *,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[TextChunk]:
    """Build searchable chunks from raw document dictionaries."""
    all_chunks: list[TextChunk] = []

    for document in documents:
        document_id = str(document["document_id"])
        text = str(document.get("content", ""))

        all_chunks.extend(
            split_text_into_chunks(
                document_id=document_id,
                text=text,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    return all_chunks


def build_search_pipeline(
    documents: list[dict],
    *,
    chunk_size: int = 300,
    overlap: int = 50,
    top_k: int = 3,
):
    """
    Build a callable retrieval pipeline.

    The returned function first reuses prebuilt chunks, then performs keyword
    search for each user query. This mirrors the basic idea of an ingestion
    stage followed by a retrieval stage.
    """
    chunks = build_chunks_from_documents(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    def run(query: str) -> list[dict]:
        return search_documents(
            query=query,
            chunks=chunks,
            top_k=top_k,
        )

    return run
