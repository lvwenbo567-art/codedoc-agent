from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from chunker import chunk_text, build_chunks

def test_chunk_text_basic():
    text="abcdefghij"

    chunks=chunk_text(
        text=text,
        chunk_size=5,
        overlap=2,
    )

    assert chunks == ["abcde", "defgh", "ghij"]

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
    assert chunk["chunk_type"] == "document"