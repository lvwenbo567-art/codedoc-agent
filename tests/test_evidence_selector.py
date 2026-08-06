from context_engineering.evidence_selector import select_evidence
from context_engineering.token_counter import CharacterTokenCounter


def test_evidence_selector_deduplicates_limits_source_and_truncates() -> None:
    result = select_evidence(
        evidence_items=[
            {"chunk_id": "a", "content_hash": "h1", "source_path": "a.py", "content": "abcdef", "score": 9},
            {"chunk_id": "a", "content_hash": "h2", "source_path": "a.py", "content": "other", "score": 8},
            {"chunk_id": "b", "content_hash": "h1", "source_path": "b.py", "content": "other", "score": 7},
            {"chunk_id": "c", "content_hash": "h3", "source_path": "a.py", "content": "long text", "score": 6},
        ],
        token_counter=CharacterTokenCounter(),
        max_total_tokens=12,
        max_single_tokens=4,
        max_items=3,
        max_items_per_source=1,
    )
    assert [item["chunk_id"] for item in result.selected] == ["a"]
    assert result.selected[0]["context_truncated"] is True
