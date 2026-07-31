from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Protocol#定义“协议类型” 只要你有这个方法，就算符合这个协议

from evaluation.agent_eval_metrics import (
    evaluate_agent_case,
    summarize_agent_eval_results,
)
from evaluation.agent_eval_schema import (
    AgentEvalCase,
    AgentEvalObservation,
    AgentEvalReport,
)
from langgraph_agent.tool_agent_service import CodeDocToolAgentService


class AgentEvalExecutorProtocol(Protocol):
    """
    评测执行器协议。

    测试里可以用 Fake Executor，真实运行时可以用 Tool Agent。
    """

    async def execute(self, case: AgentEvalCase) -> AgentEvalObservation:
        ...


def extract_tool_names_from_output(output: dict) -> list[str]:
    """
    从 Tool Agent 输出中提取工具调用顺序。
    """
    names: list[str] = []

    for item in output.get("tool_call_history", []) or []:
        tool_name = item.get("tool_name")

        if isinstance(tool_name, str) and tool_name.strip():
            names.append(tool_name.strip())

    if names:
        return names

    for message in output.get("message_trace", []) or []:
        for tool_call in message.get("tool_calls", []) or []:
            tool_name = tool_call.get("name")

            if isinstance(tool_name, str) and tool_name.strip():
                names.append(tool_name.strip())

    return names


class ToolAgentEvalExecutor:
    """
    使用 Day37 Tool Agent 执行评测 Case。
    """

    def __init__(
        self,
        *,
        service: CodeDocToolAgentService,
        recursion_limit: int = 30,
    ) -> None:
        self.service = service
        self.recursion_limit = recursion_limit

    async def execute(self, case: AgentEvalCase) -> AgentEvalObservation:
        started_at = time.perf_counter()

        try:
            output = await self.service.arun(
                query=case.query,
                project_id=case.project_id,
                recursion_limit=self.recursion_limit,
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - started_at) * 1000, 2)

            return AgentEvalObservation(
                status="failed",
                answer="",
                tool_names=[],
                stop_reason="execution_error",
                latency_ms=latency_ms,
                error_message=str(exc),
                raw_output={},
            )

        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        stop_reason = str(output.get("stop_reason") or "unknown")

        if stop_reason == "interrupted":
            status = "interrupted"
        elif output.get("completed") is True:
            status = "completed"
        else:
            status = "failed"

        return AgentEvalObservation(
            status=status,
            answer=str(output.get("answer") or ""),
            tool_names=extract_tool_names_from_output(output),
            stop_reason=stop_reason,
            latency_ms=latency_ms,
            error_message=output.get("error_message"),
            raw_output=output,
        )


async def run_agent_evaluation(
    *,
    cases: list[AgentEvalCase],
    executor: AgentEvalExecutorProtocol,
    dataset_path: str,
    model_provider: str,
    model_name: str,
) -> AgentEvalReport:
    """
    执行整批 Agent Evaluation，并生成报告对象。
    """
    results = []

    for case in cases:
        observation = await executor.execute(case)
        results.append(
            evaluate_agent_case(
                case=case,
                observation=observation,
            )
        )

    return AgentEvalReport(
        evaluation_id=f"eval_{uuid.uuid4().hex}",
        generated_at=datetime.now(timezone.utc).isoformat(),
        dataset_path=dataset_path,
        model_provider=model_provider,
        model_name=model_name,
        summary=summarize_agent_eval_results(results),
        results=results,
    )
