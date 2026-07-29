from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps
from typing_extensions import TypedDict


ToolAgentStopReason = Literal[
    "running",
    "completed",
    "model_call_limit",
    "tool_call_limit",
    "repeated_tool_call",
    "invalid_tool_call",#模型请求了未注册工具
    "remaining_steps_limit",#LangGraph 剩余步数不足
    "empty_model_response",#模型既没有返回文本，也没有返回 tool_calls
    "model_execution_error",#模型调用失败
    "graph_recursion_limit",#LangGraph 最终 recursion_limit 被触发
    "execution_error",#其他未知执行错误
]


class ToolCallHistoryItem(TypedDict, total=False):
    '''单次工具调用历史记录'''
    sequence: int#表示这是第几次工具调用
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    signature: str#工具名 + 排序后的参数 JSON search_code:{"query":"RerankClient","top_k":5}
    #这样可以判断模型是不是一直调用同一个工具和同一组参数。
    repeat_index: int
    model_call_index: int#表示这个工具调用是第几次模型调用产生的


class CodeDocToolAgentState(TypedDict, total=False):
    """
    Day37 Tool Agent 共享状态。

    messages 使用 add_messages，让 HumanMessage、AIMessage、ToolMessage
    能被 LangGraph 正确追加和反序列化。
    """

    query: str
    project_id: int
    messages: Annotated[list[BaseMessage], add_messages]
    model_call_count: int
    tool_call_count: int
    max_model_calls: int
    max_tool_calls: int
    max_identical_tool_calls: int
    tool_call_history: Annotated[list[ToolCallHistoryItem], operator.add]
    execution_steps: Annotated[list[str], operator.add]
    remaining_steps: RemainingSteps
    answer: str
    completed: bool
    stop_reason: ToolAgentStopReason
    error_message: str | None
