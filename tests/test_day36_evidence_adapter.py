from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from langgraph_agent.evidence_adapter import (
    compress_project_structure_evidence,
    convert_item_to_evidence,
    convert_result_to_evidence,
    evidence_to_answer_chunks,
)


def test_rerank_score_has_priority():
    evidence = convert_item_to_evidence(
        item={
            "chunk_id": "c1",
            "source_path": "app/a.py",
            "content": "code",
            "final_score": 0.2,
            "rerank_score": 0.9,
        },
        evidence_type="code_search",
    )

    assert evidence is not None
    assert evidence["score"] == 0.9


def test_empty_content_is_filtered():
    evidence = convert_item_to_evidence(
        item={
            "content": "",
        },
        evidence_type="code_search",
    )

    assert evidence is None


def test_result_list_converts_to_evidence():
    evidence = convert_result_to_evidence(
        data={
            "results": [
                {
                    "chunk_id": "c1",
                    "source_path": "app/a.py",
                    "content": "def a(): pass",
                }
            ]
        },
        evidence_type="code_search",
    )

    assert evidence[0]["chunk_id"] == "c1"


def test_project_structure_entries_are_compressed():
    evidence = convert_result_to_evidence(
        data={
            "entries": [
                {"path": "app", "type": "directory", "content": "app"},
                {"path": "app/api.py", "type": "file", "content": "api"},
            ]
        },
        evidence_type="project_structure",
    )

    compressed = compress_project_structure_evidence(evidence)

    assert len(compressed) == 1
    assert compressed[0]["metadata"]["original_entry_count"] == 2


def test_answer_chunks_fill_required_fields():
    chunks = evidence_to_answer_chunks(
        [
            {
                "source_path": "app/a.py",
                "evidence_type": "code_search",
                "content": "def a(): pass",
            }
        ]
    )

    assert chunks[0]["chunk_id"]
    assert chunks[0]["rank"] == 1


def test_empty_explicit_content_with_metadata_is_filtered():
    evidence = convert_item_to_evidence(
        item={
            "source_path": "app/a.py",
            "chunk_id": "c1",
            "content": "",
        },
        evidence_type="code_search",
    )

    assert evidence is None
