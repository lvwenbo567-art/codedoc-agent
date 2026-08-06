from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSecurityDecision:
    allowed: bool
    error_message: str | None = None


class ToolSecurityPolicy:
    """在 ToolNode 之前执行的第二道参数边界检查。"""

    max_query_chars = 500#任何字符串型工具参数最多 500 字符。
    max_top_k = 10#search_code / search_documents 最多召回 10 条最终结果。
    max_candidate_top_k = 50#搜索候选最多 50 条。
    max_read_lines = 1000#read_file_range 单次最多读取 1000 行。

    def validate_call(self, *, tool_name: str, arguments: dict[str, Any]) -> ToolSecurityDecision:
        if not tool_name.strip():
            return ToolSecurityDecision(False, "工具名称不能为空")
        for key, value in arguments.items():
            if isinstance(value, str) and len(value) > self.max_query_chars:
                return ToolSecurityDecision(False, f"工具参数 {key} 长度超过限制")
        if tool_name in {"search_code", "search_documents"}:
            if int(arguments.get("top_k", 5)) > self.max_top_k:
                return ToolSecurityDecision(False, "top_k 超过安全限制")
            if int(arguments.get("candidate_top_k", 20)) > self.max_candidate_top_k:
                return ToolSecurityDecision(False, "candidate_top_k 超过安全限制")
        if tool_name == "read_file_range":
            start = int(arguments.get("start_line", 1))
            end = int(arguments.get("end_line", start))
            if end - start + 1 > self.max_read_lines:
                return ToolSecurityDecision(False, "单次读取行数超过安全限制")
        return ToolSecurityDecision(True)

    def validate_calls(self, tool_calls: list[dict[str, Any]]) -> ToolSecurityDecision:
        for call in tool_calls:
            result = self.validate_call(
                tool_name=str(call.get("name") or ""),
                arguments=call.get("args") if isinstance(call.get("args"), dict) else {},
            )
            if not result.allowed:
                return result
        return ToolSecurityDecision(True)
