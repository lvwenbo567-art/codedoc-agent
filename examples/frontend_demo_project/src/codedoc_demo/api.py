"""A tiny application API for the demo retrieval project."""

from __future__ import annotations

from codedoc_demo.pipeline import build_search_pipeline


DEFAULT_DOCUMENTS = [
    {
        "document_id": "architecture",
        "content": (
            "The project turns source documents into searchable chunks. "
            "It normalizes text, splits documents into overlapping chunks, "
            "then searches those chunks with a keyword scoring function."
        ),
    },
    {
        "document_id": "usage",
        "content": (
            "Run python -m pytest tests to validate the project. "
            "The tests cover chunk splitting, keyword scoring and retrieval."
        ),
    },
]


def answer_question(query: str) -> dict:
    """Answer a question using the demo retrieval pipeline."""
    pipeline = build_search_pipeline(DEFAULT_DOCUMENTS)
    results = pipeline(query)

    if not results:
        return {
            "answer": "No matching document chunks were found.",
            "results": [],
        }

    best = results[0]

    return {
        "answer": (
            "The best matching document is "
            f"{best['document_id']} with score {best['score']}."
        ),
        "results": results,
    }
