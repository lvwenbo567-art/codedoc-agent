from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ReviewDecisionType = Literal[
    "approve",
    "reject",
    "edit",
]


class StrictReviewModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReviewToolCall(StrictReviewModel):
    '''这个表示“一个被审批的工具调用”
    id：工具调用 ID
    name：工具名称
    args：工具参数
    '''
    id: str = Field(min_length=1, max_length=300)
    name: str = Field(min_length=1, max_length=200)
    args: dict[str, Any] = Field(default_factory=dict)


class HumanReviewDecision(StrictReviewModel):
    decision: ReviewDecisionType
    feedback: str | None = Field(default=None, max_length=2000)
    edited_tool_calls: list[ReviewToolCall] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_decision(self) -> "HumanReviewDecision":
        if self.decision == "edit" and not self.edited_tool_calls:
            raise ValueError("decision=edit 时必须提供 edited_tool_calls")

        if self.decision != "edit" and self.edited_tool_calls:
            raise ValueError(
                "只有 decision=edit 时才允许提供 edited_tool_calls"
            )

        return self


class HumanReviewHistoryItem(StrictReviewModel):
    request_id: str
    decision: ReviewDecisionType
    feedback: str | None = None
    original_tool_calls: list[ReviewToolCall] = Field(default_factory=list)
    final_tool_calls: list[ReviewToolCall] = Field(default_factory=list)
