from __future__ import annotations
import uuid
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig#这是 LangChain / LangGraph 的运行配置类型。
from langgraph.errors import GraphInterrupt
#interrupt(payload) 第一次触发中断时，LangGraph 底层可能通过 GraphInterrupt 控制图暂停。
from langgraph.types import Command, interrupt

from langgraph_agent.human_review_schema import HumanReviewDecision
from langgraph_agent.tool_agent_dependencies import CodeDocToolAgentDependencies
from langgraph_agent.tool_agent_nodes import (
    CodeDocToolAgentNodes,
    _last_ai_message,
    redact_tool_messages,
)
from langgraph_agent.tool_agent_state import CodeDocToolAgentState
from langgraph_agent.tool_call_guard import evaluate_tool_calls
from langgraph_agent.tool_call_normalizer import normalize_tool_calls
from security.tool_security_policy import ToolSecurityPolicy


def _tool_call_ids(tool_calls: list[dict[str, Any]]) -> set[str]:
    return {str(call.get("id") or "") for call in tool_calls}


def _review_history_item(
    *,
    request_id: str,
    decision: HumanReviewDecision,
    original_tool_calls: list[dict[str, Any]],
    final_tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "decision": decision.decision,
        "feedback": decision.feedback,
        "original_tool_calls": original_tool_calls,
        "final_tool_calls": final_tool_calls,
    }


class HumanReviewToolAgentNodes(CodeDocToolAgentNodes):
    """
    在 Day37 Tool Agent 节点基础上增加人工审批节点。
    """

    def __init__(
        self,
        *,
        dependencies: CodeDocToolAgentDependencies,
        tool_node: Any | None = None,
    ) -> None:
        super().__init__(dependencies=dependencies, tool_node=tool_node)

    def _requires_review(self, tool_calls: list[dict[str, Any]]) -> bool:
        '''当前这批工具调用是否需要人工审批'''
        runtime = self.dependencies.runtime

        if not runtime.enable_human_review:
            return False

        protected_tools = set(runtime.approval_required_tools)

        return any(
            str(call.get("name") or "") in protected_tools
            for call in tool_calls
        )

    def controller_node(
        self,
        state: CodeDocToolAgentState,
    ) -> Command[
        Literal[
            "human_review",
            "prepare_tools",
            "finalize",
            "limit_answer",
        ]
    ]:
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

        tool_calls = normalize_tool_calls(list(last_ai_message.tool_calls or []))

        if not tool_calls:
            answer_text = str(getattr(last_ai_message, "content", "") or "").strip()

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
                    "execution_steps": ["controller_blocked"],
                },
            )

        if self._requires_review(tool_calls):
            return Command(
                goto="human_review",
                update={
                    "pending_tool_calls": tool_calls,
                    "approval_request_id": str(uuid.uuid4()),
                    "approval_status": "pending",
                    "stop_reason": "interrupted",
                    "execution_steps": ["controller_review"],
                },
            )

        security_result = ToolSecurityPolicy().validate_calls(tool_calls)
        if not security_result.allowed:
            return Command(
                goto="limit_answer",
                update={
                    "stop_reason": "invalid_tool_call",
                    "error_message": security_result.error_message,
                    "execution_steps": ["controller_security_blocked"],
                },
            )

        return Command(
            goto="prepare_tools",
            update={
                "messages": [
                    last_ai_message.model_copy(
                        update={"tool_calls": tool_calls}
                    )
                ],
                "pending_tool_calls": tool_calls,
                "approval_request_id": None,
                "approval_status": "not_required",
                "execution_steps": ["controller_prepare_tools"],
            },
        )

    def human_review_node(
        self,
        state: CodeDocToolAgentState,
    ) -> Command[Literal["prepare_tools", "agent", "limit_answer"]]:
        """
        暂停图执行，等待人类对工具调用 approve / reject / edit。

        interrupt 前只构造 payload，不执行真实工具，也不写外部副作用。
        """
        pending_tool_calls = list(state.get("pending_tool_calls") or [])
        request_id = str(state.get("approval_request_id") or "")

        payload = {
            "type": "tool_approval",
            "request_id": request_id,
            "project_id": state.get("project_id"),
            "thread_id": state.get("thread_id"),
            "effective_thread_id": state.get("effective_thread_id"),
            "query": state.get("query"),
            "tool_calls": pending_tool_calls,
            "instructions": (
                "请返回 decision=approve/reject/edit；edit 时必须保持 "
                "tool_call id 不变，只能修改已注册工具的参数。"
            ),
        }

        try:
            raw_decision = interrupt(payload)
            '''
            第一次运行到这里时：
            Graph 暂停
            payload 返回给 API
            checkpoint 保存当前状态
            这时函数不会继续往下走。
            等用户调用 resume：
            Command(resume={"decision": "approve"})
            Graph 会从这个节点恢复。
            恢复时：
            interrupt(payload)
            会返回 resume 传进来的 decision。
            这就是 LangGraph HITL 的核心机制。   
            '''
        except GraphInterrupt:
            raise#interrupt() 的底层机制可能会抛出 LangGraph 特殊异常来控制中断
        #这是正常中断，让 LangGraph 自己处理
        except Exception as exc:
            return Command(
                goto="limit_answer",
                update={
                    "stop_reason": "invalid_review_decision",
                    "error_message": f"人工审批中断失败：{exc}",
                    "execution_steps": ["human_review_invalid"],
                },
            )

        try:
            decision = HumanReviewDecision.model_validate(raw_decision)
        except Exception as exc:
            return Command(
                goto="limit_answer",
                update={
                    "stop_reason": "invalid_review_decision",
                    "error_message": f"人工审批结果不合法：{exc}",
                    "execution_steps": ["human_review_invalid"],
                },
            )

        if decision.decision == "approve":
            return Command(
                goto="prepare_tools",
                update={
                    "approval_status": "approved",
                    "stop_reason": "running",
                    "review_history": [
                        _review_history_item(
                            request_id=request_id,
                            decision=decision,
                            original_tool_calls=pending_tool_calls,
                            final_tool_calls=pending_tool_calls,
                        )
                    ],
                    "execution_steps": ["human_review_approved"],
                },
            )

        if decision.decision == "reject":
            feedback = decision.feedback or "用户拒绝执行该工具调用。"
            tool_messages = [
                ToolMessage(
                    content=f"工具调用被人工拒绝：{feedback}",
                    tool_call_id=str(call.get("id") or ""),
                    name=str(call.get("name") or ""),
                )
                for call in pending_tool_calls
            ]

            return Command(
                goto="agent",
                update={
                    "messages": tool_messages,
                    "pending_tool_calls": [],
                    "approval_request_id": None,
                    "approval_status": "rejected",
                    "stop_reason": "running",
                    "review_history": [
                        _review_history_item(
                            request_id=request_id,
                            decision=decision,
                            original_tool_calls=pending_tool_calls,
                            final_tool_calls=[],
                        )
                    ],
                    "execution_steps": ["human_review_rejected"],
                },
            )

        edited_tool_calls = [
            call.model_dump()
            for call in decision.edited_tool_calls
        ]

        if _tool_call_ids(edited_tool_calls) != _tool_call_ids(pending_tool_calls):
            return Command(
                goto="limit_answer",
                update={
                    "stop_reason": "invalid_review_decision",
                    "error_message": "edit 必须保持原 tool_call id 集合不变。",
                    "execution_steps": ["human_review_invalid"],
                },
            )

        invalid_tools = [
            call["name"]
            for call in edited_tool_calls
            if call["name"] not in self.dependencies.allowed_tool_names
        ]

        if invalid_tools:
            return Command(
                goto="limit_answer",
                update={
                    "stop_reason": "invalid_review_decision",
                    "error_message": (
                        "edit 使用了未注册工具："
                        + ", ".join(sorted(set(invalid_tools)))
                    ),
                    "execution_steps": ["human_review_invalid"],
                },
            )

        messages = list(state.get("messages") or [])
        last_ai_message = _last_ai_message(messages)

        if last_ai_message is None:
            return Command(
                goto="limit_answer",
                update={
                    "stop_reason": "invalid_review_decision",
                    "error_message": "edit 时未找到原始 AIMessage。",
                    "execution_steps": ["human_review_invalid"],
                },
            )

        edited_message = last_ai_message.model_copy(
            update={"tool_calls": edited_tool_calls}
        )

        return Command(
            goto="prepare_tools",
            update={
                "messages": [edited_message],
                "pending_tool_calls": edited_tool_calls,
                "approval_request_id": None,
                "approval_status": "edited",
                "stop_reason": "running",
                "review_history": [
                    _review_history_item(
                        request_id=request_id,
                        decision=decision,
                        original_tool_calls=pending_tool_calls,
                        final_tool_calls=edited_tool_calls,
                    )
                ],
                "execution_steps": ["human_review_edited"],
            },
        )

    def prepare_tools_node(
        self,
        state: CodeDocToolAgentState,
    ) -> Command[Literal["tools", "limit_answer"]]:
        pending_tool_calls = list(state.get("pending_tool_calls") or [])

        if not pending_tool_calls:
            return Command(
                goto="limit_answer",
                update={
                    "stop_reason": "invalid_tool_call",
                    "error_message": "没有待执行的工具调用。",
                    "execution_steps": ["prepare_tools_blocked"],
                },
            )

        guard_result = evaluate_tool_calls(
            state=state,
            tool_calls=pending_tool_calls,
            allowed_tool_names=self.dependencies.allowed_tool_names,
        )

        if not guard_result.allowed:
            return Command(
                goto="limit_answer",
                update={
                    "stop_reason": guard_result.stop_reason,
                    "error_message": guard_result.error_message,
                    "execution_steps": ["prepare_tools_blocked"],
                },
            )

        security_result = ToolSecurityPolicy().validate_calls(
            pending_tool_calls
        )
        if not security_result.allowed:
            return Command(
                goto="limit_answer",
                update={
                    "stop_reason": "invalid_tool_call",
                    "error_message": security_result.error_message,
                    "execution_steps": ["prepare_tools_security_blocked"],
                },
            )

        messages = list(state.get("messages") or [])
        last_ai_message = _last_ai_message(messages)

        if last_ai_message is None:
            return Command(
                goto="limit_answer",
                update={
                    "stop_reason": "invalid_tool_call",
                    "error_message": "准备执行工具时未找到原始 AIMessage。",
                    "execution_steps": ["prepare_tools_blocked"],
                },
            )

        return Command(
            goto="tools",
            update={
                "messages": [
                    last_ai_message.model_copy(
                        update={"tool_calls": pending_tool_calls}
                    )
                ],
                "tool_call_count": (
                    int(state.get("tool_call_count", 0))
                    + len(pending_tool_calls)
                ),
                "tool_call_history": guard_result.history_items,
                "pending_tool_calls": [],
                "execution_steps": ["prepare_tools"],
            },
        )

    def execute_tools_node(
        self,
        state: CodeDocToolAgentState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        try:
            result = self.tool_node.invoke(state, config=config)
        except Exception as exc:
            return {
                "stop_reason": "execution_error",
                "error_message": f"ToolNode 执行失败：{exc}",
                "execution_steps": ["tools"],
            }

        if isinstance(result, dict):
            return {
                **result,
                "messages": redact_tool_messages(
                    list(result.get("messages") or [])
                ),
                "execution_steps": ["tools"],
            }

        return {
            "messages": redact_tool_messages(list(result or [])),
            "execution_steps": ["tools"],
        }
