from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1]
        / "app"
    )
)

from langgraph_agent.reducers import merge_evidence


def test_merge_evidence_deduplicates_by_chunk_id():
    current = [
        {
            "chunk_id": "chunk-1",
            "content": "old",
        }
    ]
    new = [
        {
            "chunk_id": "chunk-1",
            "content": "new",
        },
        {
            "chunk_id": "chunk-2",
            "content": "another",
        },
    ]

    merged = merge_evidence(
        current,
        new,
    )

    assert len(merged) == 2
    assert merged[0]["content"] == "old"
    assert merged[1]["chunk_id"] == "chunk-2"


def test_merge_evidence_deduplicates_structure_content():
    evidence = {
        "source_path": "<project-structure>",
        "evidence_type": "project_structure",
        "content": "app tests",
    }

    merged = merge_evidence(
        [evidence],
        [dict(evidence)],
    )

    assert merged == [evidence]


def test_merge_evidence_deduplicates_by_qualified_name():
    current = [
        {
            "source_path": "app/services/rerank_service.py",
            "qualified_name": "RerankService.score",
            "content": "old symbol evidence",
        }
    ]
    new = [
        {
            "source_path": "app/services/rerank_service.py",
            "qualified_name": "RerankService.score",
            "content": "new symbol evidence",
        }
    ]

    merged = merge_evidence(
        current,
        new,
    )

    assert len(merged) == 1
    assert merged[0]["content"] == "old symbol evidence"


def test_merge_evidence_ignores_non_dict_items():
    merged = merge_evidence(
        [{"chunk_id": "chunk-1"}],
        ["bad", {"chunk_id": "chunk-2"}],
    )

    assert [
        item["chunk_id"]
        for item in merged
    ] == [
        "chunk-1",
        "chunk-2",
    ]
