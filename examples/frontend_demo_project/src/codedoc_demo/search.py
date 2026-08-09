"""Keyword search utilities for the demo project."""

from __future__ import annotations

from codedoc_demo.parser import TextChunk


def keyword_score(query: str, text: str) -> int:
    """
    Count how many query terms appear in the target text.

    The function is intentionally simple so it is easy to locate and explain
    during CodeDoc Research Agent frontend testing.
    """
    terms = [
        term.lower()
        for term in query.split()
        if term.strip()
    ]

    lower_text = text.lower()

    return sum(
        lower_text.count(term)
        for term in terms
    )


def search_documents(
    *,
    query: str,
    chunks: list[TextChunk],
    top_k: int = 3,
) -> list[dict]:
    """Search chunks by keyword score and return Top-K matches."""
    scored_results: list[dict] = []

    for chunk in chunks:
        score = keyword_score(
            query=query,
            text=chunk.content,
        )

        if score <= 0:
            continue

        scored_results.append(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "score": score,
                "content": chunk.content,
                "start": chunk.start,
                "end": chunk.end,
            }
        )

    scored_results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored_results[:top_k]
