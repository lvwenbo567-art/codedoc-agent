from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from prompt_builder import (
    SYSTEM_PROMPT,
    build_context,
    build_rag_messages,
    build_rag_prompt,
    build_rag_user_prompt,
)


def build_chunks():
    return [
        {
            "rank": 1,
            "chunk_id": "main.py::chunk_0",
            "source_path": "main.py",
            "source_name": "main.py",
            "source_suffix": ".py",
            "chunk_type": "code",
            "chunk_index": 0,
            "content": "def main(): pass",
            "score": 1.0,
        },
        {
            "rank": 2,
            "chunk_id": "README.md::chunk_0",
            "source_path": "README.md",
            "source_name": "README.md",
            "source_suffix": ".md",
            "chunk_type": "document",
            "chunk_index": 0,
            "content": "This project explains CodeDoc.",
            "score": 0.5,
        },
    ]


def test_build_context_includes_sources_and_content():
    context = build_context(build_chunks())

    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert "main.py" in context
    assert "README.md" in context
    assert "def main(): pass" in context


def test_build_context_skips_chunks_without_content():
    chunks = build_chunks()
    chunks[0]["content"] = "   "

    context = build_context(chunks)

    assert "main.py::chunk_0" not in context
    assert "README.md::chunk_0" in context


def test_build_context_respects_max_context_chars():
    context = build_context(
        retrieved_chunks=build_chunks(),
        max_context_chars=40,
    )

    assert len(context) <= 40


def test_build_context_invalid_max_context_chars():
    with pytest.raises(ValueError):
        build_context(
            retrieved_chunks=build_chunks(),
            max_context_chars=0,
        )


def test_build_rag_prompt_includes_query_context_and_answer_marker():
    prompt = build_rag_prompt(
        query="main function?",
        retrieved_chunks=build_chunks(),
    )

    assert "CodeDoc Research Agent" in prompt
    assert "main function?" in prompt
    assert "[Source 1]" in prompt
    assert "def main(): pass" in prompt


def test_build_rag_user_prompt_includes_query_and_context():
    prompt = build_rag_user_prompt(
        query="main function?",
        retrieved_chunks=build_chunks(),
    )

    assert "main function?" in prompt
    assert "[Source 1]" in prompt
    assert "def main(): pass" in prompt


def test_build_rag_messages_uses_system_and_user_roles():
    messages = build_rag_messages(
        query="main function?",
        retrieved_chunks=build_chunks(),
    )

    assert messages == [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": messages[1]["content"],
        },
    ]
    assert "main function?" in messages[1]["content"]
    assert "[Source 1]" in messages[1]["content"]


def test_build_rag_prompt_empty_query():
    with pytest.raises(ValueError):
        build_rag_prompt(
            query="   ",
            retrieved_chunks=build_chunks(),
        )


def test_build_rag_prompt_without_retrieved_content():
    prompt = build_rag_prompt(
        query="main function?",
        retrieved_chunks=[],
    )

    assert "main function?" in prompt
    assert "main.py::chunk_0" not in prompt
    assert "def main(): pass" not in prompt
