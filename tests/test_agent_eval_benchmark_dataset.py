from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "app"))

from evaluation.agent_eval_dataset import load_agent_eval_cases


def test_packaged_agent_benchmark_has_balanced_case_coverage() -> None:
    """发布用 Agent 基准集应保持可解释的规模和任务覆盖。"""
    cases = load_agent_eval_cases(
        str(ROOT / "data/evaluation/codedoc_agent_eval.jsonl")
    )

    assert 30 <= len(cases) <= 50

    tags = {
        tag
        for case in cases
        for tag in case.tags
    }
    assert {
        "symbol",
        "file_read",
        "structure",
        "document",
        "workflow",
        "test",
        "safety",
    }.issubset(tags)
