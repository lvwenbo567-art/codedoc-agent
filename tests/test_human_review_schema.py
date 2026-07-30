from __future__ import annotations

from pathlib import Path
import sys

import pytest
from pydantic import ValidationError


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from langgraph_agent.human_review_schema import HumanReviewDecision


def test_approve_does_not_need_edited_tool_calls() -> None:
    decision = HumanReviewDecision(decision="approve")

    assert decision.decision == "approve"
    assert decision.edited_tool_calls == []


def test_reject_allows_feedback() -> None:
    decision = HumanReviewDecision(
        decision="reject",
        feedback="这次不允许读取源码",
    )

    assert decision.feedback == "这次不允许读取源码"


def test_edit_requires_edited_tool_calls() -> None:
    with pytest.raises(ValidationError):
        HumanReviewDecision(decision="edit")


def test_approve_forbids_edited_tool_calls() -> None:
    with pytest.raises(ValidationError):
        HumanReviewDecision(
            decision="approve",
            edited_tool_calls=[
                {
                    "id": "call_1",
                    "name": "read_file_range",
                    "args": {},
                }
            ],
        )


def test_empty_tool_call_id_fails_validation() -> None:
    with pytest.raises(ValidationError):
        HumanReviewDecision(
            decision="edit",
            edited_tool_calls=[
                {
                    "id": "",
                    "name": "read_file_range",
                    "args": {},
                }
            ],
        )
