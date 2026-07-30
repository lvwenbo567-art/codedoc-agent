from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.prebuilt import ToolNode#ToolNode 是 LangGraph 已经封装好的“工具执行节点”。
from langgraph.types import Command, Overwrite#更新哪些状态，以及下一步去哪个节点。

from langchain_agent.chat_service import extract_message_text
from langchain_agent.message_window_middleware import select_message_window
from langgraph_agent.tool_agent_dependencies import CodeDocToolAgentDependencies
from langgraph_agent.tool_agent_state import CodeDocToolAgentState
from langgraph_agent.tool_call_guard import evaluate_tool_calls


CODEDOC_TOOL_AGENT_SYSTEM_PROMPT = """
你是 CodeDoc Research Agent。

你的任务是基于当前项目中的真实代码、文档和目录结构回答用户问题。

系统已经通过工具 schema 提供了可用工具的名称、参数和基础描述；
你应根据用户问题选择合适工具，不要编造不存在的工具。

工具使用策略：
1. 明确函数名、类名或方法名的问题，优先使用精确符号定义查询。
2. 如果工具结果提供了 source_path、start_line、end_line，且用户需要源码解释，
   应读取对应文件范围后再回答。
3. 项目目录、模块、入口文件问题，优先查看项目结构。
4. README、启动、配置、部署和说明类问题，优先检索项目文档。
5. 代码实现、调用关系、检索流程、RAG 流程问题，优先检索代码。
6. 工具返回错误时，可以换一种工具或参数继续有限尝试。
7. 已经拿到足够证据时，应停止调用工具并直接给出最终答案。
8. 不要重复调用相同工具和相同参数。
9. 不要编造项目中不存在的文件、函数或实现。
10. 不要尝试访问项目根目录之外的文件。
11. 如果证据不足，请明确说明证据不足。
""".strip()


ControllerGoto = Literal["tools", "finalize", "limit_answer"]


def _last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message

    return None


def _message_preview(message: BaseMessage, max_chars: int) -> dict[str, Any]:
    content = str(getattr(message, "content", ""))
    '''
    这个函数用于生成 trace。
    它把 LangChain Message 转成可 JSON 返回的 dict。
    '''
    if len(content) > max_chars:
        content = content[:max_chars] + "\n...[content truncated]"

    return {
        "type": message.__class__.__name__,
        "content": content,
        "tool_call_id": getattr(message, "tool_call_id", None),
        "tool_calls": list(getattr(message, "tool_calls", []) or []),
    }


@dataclass
class CodeDocToolAgentNodes:
    dependencies: CodeDocToolAgentDependencies
    tool_node: Any | None = None

    def __post_init__(self) -> None:
        if self.tool_node is None:
            self.tool_node = ToolNode(self.dependencies.tools)
            '''
            读取 AIMessage.tool_calls
            找到对应 LangChain Tool
            执行工具
            返回 ToolMessage
            '''
    def initialize_node(self, state: CodeDocToolAgentState) -> dict:
        query = str(state.get("query") or "").strip()
        previous_turn_index = int(state.get("turn_index", 0))

        return {
            "query": query,
            "turn_index": previous_turn_index + 1,
            "model_call_count": 0,
            "tool_call_count": 0,
            "max_model_calls": self.dependencies.runtime.max_model_calls,
            "max_tool_calls": self.dependencies.runtime.max_tool_calls,
            "max_identical_tool_calls": (
                self.dependencies.runtime.max_identical_tool_calls
            ),
            "tool_call_history": Overwrite(value=[]),
            "pending_tool_calls": [],
            "approval_request_id": None,
            "approval_status": "not_required",
            "review_history": Overwrite(value=[]),
            "answer": "",
            "completed": False,
            "stop_reason": "running",
            "error_message": None,
            "execution_steps": Overwrite(value=["initialize"]),
        }

    def call_model_node(self, state: CodeDocToolAgentState) -> dict:
        model_call_count = int(state.get("model_call_count", 0))
        max_model_calls = int(state.get("max_model_calls", 6))

        if model_call_count >= max_model_calls:
            return {
                "stop_reason": "model_call_limit",
                "error_message": f"模型调用次数达到上限 {max_model_calls}",
                "execution_steps": ["model_call_limit"],
            }

        remaining_steps = state.get("remaining_steps")

        if isinstance(remaining_steps, int) and remaining_steps <= 2:
            return {
                "stop_reason": "remaining_steps_limit",
                "error_message": "LangGraph 剩余步数不足，主动停止。",
                "execution_steps": ["remaining_steps_limit"],
            }

        messages = list(state.get("messages") or [])
        selected_messages = select_message_window(
            messages=messages,
            max_messages=self.dependencies.runtime.max_model_messages,
        )
        model_messages = [
            SystemMessage(content=CODEDOC_TOOL_AGENT_SYSTEM_PROMPT),
            *selected_messages,
        ]

        try:
            response = self.dependencies.model_with_tools.invoke(model_messages)
        except Exception as exc:
            return {
                "stop_reason": "model_execution_error",
                "error_message": f"模型调用失败：{exc}",
                "execution_steps": ["model_call"],
            }

        if isinstance(response, str):
            response = AIMessage(content=response)

        if not isinstance(response, AIMessage):
            return {
                "stop_reason": "model_execution_error",
                "error_message": (
                    "模型返回类型不是 AIMessage："
                    f"{type(response).__name__}"
                ),
                "execution_steps": ["model_call"],
            }

        return {
            "messages": [response],
            "model_call_count": model_call_count + 1,
            "execution_steps": ["model_call"],
        }

    def controller_node(
        self,
        state: CodeDocToolAgentState,
    ) -> Command[ControllerGoto]:
        stop_reason = state.get("stop_reason")

        if stop_reason and stop_reason != "running":
            return Command(goto="limit_answer")

        messages = list(state.get("messages") or [])
        last_ai_message = _last_ai_message(messages)

        if last_ai_message is None:
            return Command(
                goto="limit_answer",
                update={
                    "stop_reason": "empty_model_response",
                    "error_message": "模型没有返回 AIMessage。",
                    "execution_steps": ["controller_limit"],
                },
            )

        tool_calls = list(last_ai_message.tool_calls or [])
        answer_text = extract_message_text(last_ai_message)

        if not tool_calls:
            if not answer_text:
                return Command(
                    goto="limit_answer",
                    update={
                        "stop_reason": "empty_model_response",
                        "error_message": "模型返回空文本且没有 tool_calls。",
                        "execution_steps": ["controller_limit"],
                    },
                )

            return Command(
                goto="finalize",
                update={"execution_steps": ["controller_finalize"]},
            )

        guard_result = evaluate_tool_calls(
            state=state,
            tool_calls=tool_calls,
            allowed_tool_names=self.dependencies.allowed_tool_names,
        )

        if not guard_result.allowed:
            return Command(
                goto="limit_answer",
                update={
                    "stop_reason": guard_result.stop_reason,
                    "error_message": guard_result.error_message,
                    "execution_steps": ["controller_limit"],
                },
            )

        return Command(
            goto="tools",
            update={
                "tool_call_count": (
                    int(state.get("tool_call_count", 0))
                    + len(tool_calls)
                ),
                "tool_call_history": guard_result.history_items,
                "execution_steps": ["controller_tools"],
            },
        )

    def tools_node(self, state: CodeDocToolAgentState) -> dict:
        try:
            result = self.tool_node.invoke(state)
        except Exception as exc:
            return {
                "stop_reason": "execution_error",
                "error_message": f"ToolNode 执行失败：{exc}",
                "execution_steps": ["tools"],
            }

        if isinstance(result, dict):
            return {
                **result,
                "execution_steps": ["tools"],
            }

        return {
            "messages": list(result or []),
            "execution_steps": ["tools"],
        }

    def finalize_node(self, state: CodeDocToolAgentState) -> dict:
        messages = list(state.get("messages") or [])
        last_ai_message = _last_ai_message(messages)
        answer = extract_message_text(last_ai_message) if last_ai_message else ""

        if not answer:
            answer = "模型没有返回有效最终回答。"

        return {
            "answer": answer,
            "completed": True,
            "stop_reason": "completed",
            "execution_steps": ["finalize"],
        }

    def limit_answer_node(self, state: CodeDocToolAgentState) -> dict:
        stop_reason = str(state.get("stop_reason") or "execution_error")
        error_message = state.get("error_message")

        reason_to_answer = {
            "model_call_limit": "Agent 达到模型调用次数上限，已安全停止。",
            "tool_call_limit": "Agent 达到工具调用次数上限，已安全停止。",
            "repeated_tool_call": "Agent 检测到重复工具调用，已安全停止。",
            "invalid_tool_call": "Agent 请求了未注册工具，已安全停止。",
            "invalid_review_decision": "Agent 审批决定不合法，已安全停止。",
            "remaining_steps_limit": "Agent 剩余图执行步数不足，已安全停止。",
            "empty_model_response": "模型没有返回有效文本或工具调用，已停止。",
            "model_execution_error": "模型调用失败，Agent 已停止。",
            "graph_recursion_limit": "Agent 达到 Graph recursion_limit，可能存在循环调用。",
            "execution_error": "Agent 执行失败，已停止。",
        }
        answer = reason_to_answer.get(stop_reason, "Agent 已安全停止。")

        if error_message:
            answer = f"{answer}原因：{error_message}"

        return {
            "answer": answer,
            "completed": True,
            "execution_steps": ["limit_answer"],
        }

    def build_message_trace(self, state: CodeDocToolAgentState) -> list[dict[str, Any]]:
        return [
            _message_preview(
                message,
                max_chars=self.dependencies.runtime.trace_content_chars,
            )
            for message in list(state.get("messages") or [])
        ]
