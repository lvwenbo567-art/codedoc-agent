from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from langgraph_agent.tool_agent_state import (
    CodeDocToolAgentState,
    ToolCallHistoryItem,
)


@dataclass(frozen=True)
class ToolCallGuardResult:
    allowed: bool
    stop_reason: str | None
    error_message: str | None#模型请求了未注册工具：delete_project
    history_items: list[ToolCallHistoryItem]
    '''
    如果工具调用允许执行，就会生成新的工具调用历史。
    这些历史会被写入：
    state["tool_call_history"]
    用于后续重复调用检查。
    '''

def normalize_tool_arguments(arguments: Any) -> dict[str, Any]:
    """
    将模型生成的 Tool 参数标准化为字典。
    """
    if isinstance(arguments, dict):
        return arguments

    return {"raw": arguments}


def build_tool_call_signature(
    *,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """
    用“工具名 + 排序后的 JSON 参数”构造重复调用签名。
    给一次工具调用生成唯一签名，用于判断重复调用。
    """
    normalized_json = json.dumps(
        arguments,
        ensure_ascii=False,#让中文保持中文，而不是变成 unicode 编码
        sort_keys=True,#让参数 key 按固定顺序排序。
        separators=(",", ":"),#去掉多余空格，让 signature 更稳定。
        default=str,
    )

    return f"{tool_name}:{normalized_json}"#search_code:{"query":"RerankClient","top_k":5}


def evaluate_tool_calls(
    *,
    state: CodeDocToolAgentState,
    tool_calls: list[dict],#模型刚刚生成的工具调用列表。
    allowed_tool_names: set[str] | frozenset[str],#允许执行的工具白名单。
) -> ToolCallGuardResult:
    """
    ToolNode 执行前的安全检查：
    1. 工具是否在白名单中；
    2. 是否超过工具调用总预算；
    3. 是否重复调用相同工具和相同参数。
    """
    current_tool_count = int(state.get("tool_call_count", 0))
    max_tool_calls = int(state.get("max_tool_calls", 10))

    if current_tool_count + len(tool_calls) > max_tool_calls:
        return ToolCallGuardResult(
            allowed=False,
            stop_reason="tool_call_limit",
            error_message=f"执行当前工具调用将超过工具调用上限 {max_tool_calls}",
            history_items=[],
        )

    previous_history = state.get("tool_call_history", [])
    signature_counts = Counter(
        str(item.get("signature") or "")
        for item in previous_history
        if item.get("signature")
    )
    '''
    每种工具调用 signature 已经出现过几次
比如历史里有：
[
    {"signature": "search_code:{\"query\":\"RerankClient\"}"},
    {"signature": "search_code:{\"query\":\"RerankClient\"}"},
    {"signature": "read_file_range:{\"source_path\":\"a.py\"}"}
]
统计结果就是：
{
    "search_code:{\"query\":\"RerankClient\"}": 2,
    "read_file_range:{\"source_path\":\"a.py\"}": 1
}
    '''
    max_identical_calls = int(state.get("max_identical_tool_calls", 2))
    model_call_index = int(state.get("model_call_count", 0))
    pending_items: list[ToolCallHistoryItem] = []

    for offset, tool_call in enumerate(tool_calls, start=1):
        tool_name = str(tool_call.get("name") or "").strip()
        tool_call_id = str(tool_call.get("id") or "").strip()

        if not tool_name or tool_name not in allowed_tool_names:
            return ToolCallGuardResult(
                allowed=False,
                stop_reason="invalid_tool_call",
                error_message=f"模型请求了未注册工具：{tool_name or '<empty>'}",
                history_items=[],
            )

        arguments = normalize_tool_arguments(tool_call.get("args", {}))
        signature = build_tool_call_signature(
            tool_name=tool_name,
            arguments=arguments,
        )
        next_repeat_index = signature_counts[signature] + 1
        #计算这是第几次相同调用
        if next_repeat_index > max_identical_calls:
            return ToolCallGuardResult(
                allowed=False,
                stop_reason="repeated_tool_call",
                error_message=(
                    "检测到相同工具与参数被重复调用："
                    f"{tool_name}，允许次数={max_identical_calls}"
                ),
                history_items=[],
            )

        signature_counts[signature] = next_repeat_index
        pending_items.append(
            ToolCallHistoryItem(
                sequence=current_tool_count + offset,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=arguments,
                signature=signature,
                repeat_index=next_repeat_index,
                model_call_index=model_call_index,
            )
        )

    return ToolCallGuardResult(
        allowed=True,
        stop_reason=None,
        error_message=None,
        history_items=pending_items,
    )
