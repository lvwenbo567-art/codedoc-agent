from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from services.keyword_retriever import extract_query_terms, score_chunk, search_chunks

def test_extract_query_terms_basic():
    terms = extract_query_terms("file loader")

    assert terms == ["file", "loader"]


def test_extract_query_terms_empty():
    with pytest.raises(ValueError):
        extract_query_terms("   ")


def test_score_chunk_content_match():
    chunk = {
        "chunk_id": "README.md::chunk_0",
        "source_path": "README.md",
        "source_name": "README.md",
        "source_suffix": ".md",
        "chunk_type": "document",
        "chunk_index": 0,
        "content": "This project uses file loader to scan files.",
        "length": 45,
    }

    score = score_chunk("file loader", chunk)

    assert score > 0

def test_score_chunk_source_name_match():
    chunk = {
        "chunk_id": "file_loader.py::chunk_0",
        "source_path": "file_loader.py",
        "source_name": "file_loader.py",
        "source_suffix": ".py",
        "chunk_type": "code",
        "chunk_index": 0,
        "content": "def scan_project_files(): pass",
        "length": 30,
    }

    score = score_chunk("file_loader", chunk)

    assert score > 0

def test_search_chunks_top_k():
    chunks = [
        {
            "chunk_id": "README.md::chunk_0",
            "source_path": "README.md",
            "source_name": "README.md",
            "source_suffix": ".md",
            "chunk_type": "document",
            "chunk_index": 0,
            "content": "This is a README file.",
            "length": 22,
        },
        {
            "chunk_id": "file_loader.py::chunk_0",
            "source_path": "file_loader.py",
            "source_name": "file_loader.py",
            "source_suffix": ".py",
            "chunk_type": "code",
            "chunk_index": 0,
            "content": "def load_project_files(): pass",
            "length": 30,
        },
        {
            "chunk_id": "chunker.py::chunk_0",
            "source_path": "chunker.py",
            "source_name": "chunker.py",
            "source_suffix": ".py",
            "chunk_type": "code",
            "chunk_index": 0,
            "content": "def chunk_text(): pass",
            "length": 22,
        },
    ]

    results = search_chunks(
        query="file",
        chunks=chunks,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]
    assert "score" in results[0]

def test_search_chunks_no_match():
    chunks = [
        {
            "chunk_id": "README.md::chunk_0",
            "source_path": "README.md",
            "source_name": "README.md",
            "source_suffix": ".md",
            "chunk_type": "document",
            "chunk_index": 0,
            "content": "hello world",
            "length": 11,
        }
    ]

    results = search_chunks(
        query="notfound",
        chunks=chunks,
        top_k=5,
    )

    assert results == []

def test_search_chunks_invalid_top_k():
    with pytest.raises(ValueError):
        search_chunks(
            query="hello",
            chunks=[],
            top_k=0,
        )
