from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from services.citation_builder import build_citations


def test_build_citations():
    chunks = [
        {
            "rank": 1,
            "chunk_id": "main.py::chunk_0",
            "source_path": "main.py",
            "source_name": "main.py",
            "source_suffix": ".py",
            "chunk_type": "code",
            "chunk_index": 0,
            "score": 0.95,
            "content": "def main(): pass",
        }
    ]

    citations = build_citations(chunks)

    assert len(citations) == 1
    assert citations[0]["citation_id"] == "Source 1"
    assert citations[0]["rank"] == 1
    assert citations[0]["chunk_id"] == "main.py::chunk_0"
    assert citations[0]["source_path"] == "main.py"
    assert citations[0]["source_name"] == "main.py"
    assert citations[0]["source_suffix"] == ".py"
    assert citations[0]["chunk_type"] == "code"
    assert citations[0]["chunk_index"] == 0
    assert citations[0]["score"] == 0.95
    assert "content" not in citations[0]


def test_build_citations_empty():
    assert build_citations([]) == []
