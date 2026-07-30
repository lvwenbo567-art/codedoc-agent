from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps
from typing_extensions import TypedDict


ToolAgentStopReason = Literal[
    "running",
    "interrupted",#被 interrupt 暂停，等待人工审批
    "completed",
    "model_call_limit",
    "tool_call_limit",
    "repeated_tool_call",
    "invalid_tool_call",
    "invalid_review_decision",
    "remaining_steps_limit",
    "empty_model_response",
    "model_execution_error",
    "graph_recursion_limit",
    "execution_error",
]


ApprovalStatus = Literal[#审批状态
    "not_required",
    "pending",
    "approved",
    "rejected",
    "edited",
]


class ToolCallHistoryItem(TypedDict, total=False):
    """
    单次工具调用历史。

    signature 用于识别“同一个工具 + 同一组参数”是否被模型重复调用。
    """

    sequence: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    signature: str
    repeat_index: int
    model_call_index: int


class ReviewHistoryItem(TypedDict, total=False):
    """
    一次人工审批历史。

    original_tool_calls 是模型原始想调用的工具；
    final_tool_calls 是审批后最终允许继续执行的工具。
    """

    request_id: str
    decision: str
    feedback: str | None
    original_tool_calls: list[dict[str, Any]]
    final_tool_calls: list[dict[str, Any]]


class CodeDocToolAgentState(TypedDict, total=False):
    """
    Tool Agent 的共享状态。

    messages 使用 add_messages，让 HumanMessage、AIMessage、ToolMessage 可以被
    LangGraph 正确追加；带相同 id 的消息会被替换，这正好支持 Day39 edit
    审批场景中替换原 AIMessage.tool_calls。
    """

    query: str
    project_id: int
    run_id: str
    thread_id: str
    effective_thread_id: str
    turn_index: int
    messages: Annotated[list[BaseMessage], add_messages]
    model_call_count: int
    tool_call_count: int
    max_model_calls: int
    max_tool_calls: int
    max_identical_tool_calls: int
    tool_call_history: Annotated[list[ToolCallHistoryItem], operator.add]
    execution_steps: Annotated[list[str], operator.add]
    pending_tool_calls: list[dict[str, Any]]#模型已经生成，但还没执行的工具调用
    approval_request_id: str | None#这一次审批请求的 ID
    approval_status: ApprovalStatus#表示当前审批状态：
    review_history: Annotated[list[ReviewHistoryItem], operator.add]#保存审批历史。
    remaining_steps: RemainingSteps
    answer: str
    completed: bool
    stop_reason: ToolAgentStopReason
    error_message: str | None
