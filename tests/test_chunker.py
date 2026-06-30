from pathlib import Path
import sys
import pytest
sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from chunker import build_chunks, chunk_text, get_chunk_type

def test_chunk_text_basic():
    text="abcdefghij"

    chunks=chunk_text(
        text=text,
        chunk_size=5,
        overlap=2,
    )

    assert chunks == ["abcde", "defgh", "ghij"]

def test_chunk_text_empty():
    chunks = chunk_text(
        text="",
        chunk_size=5,
        overlap=2,
    )

    assert chunks == []

def test_chunk_text_blank():
    chunks = chunk_text(
        text="   ",
        chunk_size=5,
        overlap=2,
    )

    assert chunks == []

def test_chunk_text_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_text(
            text="hello",
            chunk_size=0,
            overlap=0,
        )

def test_chunk_text_invalid_overlap_negative():
    with pytest.raises(ValueError):
        chunk_text(
            text="hello",
            chunk_size=5,
            overlap=-1,
        )

def test_chunk_text_invalid_overlap_equal_chunk_size():
    with pytest.raises(ValueError):
        chunk_text(
            text="hello",
            chunk_size=5,
            overlap=5,
        )

def test_chunk_text_invalid_overlap_greater_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text(
            text="hello",
            chunk_size=5,
            overlap=6,
        )

def test_get_chunk_type_code():
    assert get_chunk_type(".py") == "code"


def test_get_chunk_type_document():
    assert get_chunk_type(".md") == "document"
    assert get_chunk_type(".txt") == "document"

def test_build_chunks_type():
    files = [
        {
            "path": "test_project/README.md",
            "name": "README.md",
            "suffix": ".md",
            "content": "hello world",
            "length": 11,
        },
        {
            "path": "test_project/main.py",
            "name": "main.py",
            "suffix": ".py",
            "content": "def main():\n    pass",
            "length": 20,
        },
    ]

    chunks = build_chunks(
        files=files,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) == 2
    assert chunks[0]["chunk_type"] == "document"
    assert chunks[1]["chunk_type"] == "code"

def test_build_chunks_fields():
    files = [
        {
            "path": "test_project/README.md",
            "name": "README.md",
            "suffix": ".md",
            "content": "hello world",
            "length": 11,
        }
    ]

    chunks = build_chunks(
        files=files,
        chunk_size=100,
        overlap=20,
    )

    chunk = chunks[0]

    assert "chunk_id" in chunk
    assert "source_path" in chunk
    assert "source_name" in chunk
    assert "source_suffix" in chunk
    assert "chunk_type" in chunk
    assert "chunk_index" in chunk
    assert "content" in chunk
    assert "length" in chunk

    assert chunk["source_name"] == "README.md"
    assert chunk["source_suffix"] == ".md"
    assert chunk["chunk_type"] == "document"
    assert chunk["chunk_index"] == 0
    assert chunk["content"] == "hello world"
    assert chunk["length"] == len("hello world")

def test_build_chunks_empty_content():
    files = [
        {
            "path": "test_project/empty.md",
            "name": "empty.md",
            "suffix": ".md",
            "content": "",
            "length": 0,
        }
    ]

    chunks = build_chunks(
        files=files,
        chunk_size=100,
        overlap=20,
    )

    assert chunks == []