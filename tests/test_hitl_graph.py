from __future__ import annotations

from pathlib import Path
import asyncio
import sys

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent.human_review_graph import build_human_review_tool_agent_graph
from langgraph_agent.human_review_schema import HumanReviewDecision
from langgraph_agent.human_review_service import HumanReviewToolAgentService
from langgraph_agent.tool_agent_config import ToolAgentRuntimeConfig
from langgraph_agent.tool_agent_dependencies import CodeDocToolAgentDependencies


class FakeToolCallingModel:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1

        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "read_file_range",
                        "args": {
                            "source_path": "test_project/search.py",
                            "start_line": 1,
                            "end_line": 3,
                        },
                    }
                ],
            )

        return AIMessage(content="keyword_score 定义在 search.py。")


@tool
def read_file_range(
    source_path: str,
    start_line: int,
    end_line: int,
) -> str:
    """读取指定文件的行范围。"""
    return f"{source_path}:{start_line}-{end_line}"


def _service(model: FakeToolCallingModel) -> HumanReviewToolAgentService:
    runtime = ToolAgentRuntimeConfig(project_root=".")
    dependencies = CodeDocToolAgentDependencies(
        runtime=runtime,
        model_config=LangChainModelConfig(provider="mock"),
        model_with_tools=model,
        tools=[read_file_range],
        allowed_tool_names=frozenset({"read_file_range"}),
    )
    graph = build_human_review_tool_agent_graph(
        dependencies,
        checkpointer=InMemorySaver(),
    )
    locks: dict[str, asyncio.Lock] = {}

    def lock_provider(effective_thread_id: str) -> asyncio.Lock:
        if effective_thread_id not in locks:
            locks[effective_thread_id] = asyncio.Lock()

        return locks[effective_thread_id]

    return HumanReviewToolAgentService(
        dependencies=dependencies,
        graph=graph,
        thread_lock_provider=lock_provider,
    )


@pytest.mark.asyncio
async def test_hitl_graph_interrupt_and_resume_approve() -> None:
    model = FakeToolCallingModel()
    service = _service(model)

    interrupted = await service.start(
        query="请读取 keyword_score 附近源码",
        project_id=1,
        thread_id="hitl-test",
        recursion_limit=40,
    )

    assert interrupted["status"] == "interrupted"
    assert interrupted["interrupts"]
    assert interrupted["approval_status"] == "pending"

    resumed = await service.resume(
        project_id=1,
        thread_id="hitl-test",
        decision=HumanReviewDecision(decision="approve"),
        recursion_limit=40,
    )

    assert resumed["status"] == "completed"
    assert resumed["success"] is True
    assert resumed["approval_status"] == "approved"
    assert "tools" in resumed["execution_steps"]
