from pathlib import Path
import sys
import pytest
sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from ingestion.chunker import (
    build_chunks,
    chunk_text,
    get_chunk_type,
    is_config_aware_suffix,
)

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


def test_is_config_aware_suffix():
    assert is_config_aware_suffix(".json") is True
    assert is_config_aware_suffix(".toml") is True
    assert is_config_aware_suffix(".yaml") is True
    assert is_config_aware_suffix(".md") is False

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
    assert chunk["code_unit_type"] is None
    assert chunk["symbol_name"] is None
    assert chunk["qualified_name"] is None
    assert chunk["parent_class"] is None
    assert chunk["signature"] is None
    assert chunk["start_line"] is None
    assert chunk["end_line"] is None
    assert chunk["docstring"] == ""
    assert chunk["parser"] == "text"
    assert chunk["parse_error"] is None
    assert chunk["part_index"] == 0
    assert chunk["part_count"] == 1

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


def test_build_chunks_json_config_sections():
    files = [
        {
            "path": "test_project/package.json",
            "name": "package.json",
            "suffix": ".json",
            "content": (
                '{"scripts": {"dev": "vite"}, '
                '"dependencies": {"fastapi": "0.1.0"}}'
            ),
            "length": 72,
        }
    ]

    chunks = build_chunks(
        files=files,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) == 2
    assert {chunk["symbol_name"] for chunk in chunks} == {
        "scripts",
        "dependencies",
    }
    assert all(chunk["code_unit_type"] == "config_section" for chunk in chunks)
    assert all(chunk["parser"] == "json" for chunk in chunks)


def test_build_chunks_toml_config_sections():
    files = [
        {
            "path": "test_project/pyproject.toml",
            "name": "pyproject.toml",
            "suffix": ".toml",
            "content": (
                '[project]\nname = "codedoc"\n\n'
                '[tool.pytest.ini_options]\nasyncio_mode = "auto"\n'
            ),
            "length": 80,
        }
    ]

    chunks = build_chunks(
        files=files,
        chunk_size=120,
        overlap=20,
    )

    assert len(chunks) == 2
    assert {chunk["symbol_name"] for chunk in chunks} == {
        "project",
        "tool",
    }
    assert all(chunk["parser"] == "toml" for chunk in chunks)


def test_build_chunks_invalid_json_falls_back_to_text():
    files = [
        {
            "path": "test_project/broken.json",
            "name": "broken.json",
            "suffix": ".json",
            "content": '{"scripts": ',
            "length": 12,
        }
    ]

    chunks = build_chunks(
        files=files,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) == 1
    assert chunks[0]["parser"] == "config_fallback"
    assert chunks[0]["parse_error"]
    assert chunks[0]["content"] == '{"scripts":'
