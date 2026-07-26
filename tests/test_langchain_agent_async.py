from pathlib import Path
import asyncio
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1] / "app"
    )
)


from langchain_core.messages import AIMessage

from langchain_agent.agent_service import LangChainAgentService
from langchain_agent.model_config import LangChainModelConfig
from langchain_agent.runtime_context import CodeDocRuntimeContext


class FakeAsyncAgent:
    def __init__(self) -> None:
        self.received_context: CodeDocRuntimeContext | None = None
        self.received_thread_id: str | None = None

    async def ainvoke(self, input, config, context=None):
        self.received_context = context
        self.received_thread_id = config["configurable"]["thread_id"]

        return {
            "messages": [
                *input["messages"],
                AIMessage(content="async agent ok"),
            ],
        }


def build_async_service(tmp_path, agent):
    chunks_path = tmp_path / "chunks.json"
    index_path = tmp_path / "index.json"
    chunks_path.write_text("[]", encoding="utf-8")
    index_path.write_text("{}", encoding="utf-8")

    return LangChainAgentService(
        config=LangChainModelConfig(
            provider="openai_compatible",
            model_name="fake-model",
            base_url="http://localhost/v1",
        ),
        project_root=str(tmp_path),
        chunks_path=str(chunks_path),
        index_path=str(index_path),
        agent=agent,
    )


def test_langchain_agent_service_supports_async_runtime_context(tmp_path):
    asyncio.run(
        _run_async_agent_service_assertions(tmp_path)
    )


async def _run_async_agent_service_assertions(tmp_path):
    agent = FakeAsyncAgent()
    service = build_async_service(
        tmp_path=tmp_path,
        agent=agent,
    )

    result = await service.arun(
        query="hello async agent",
        project_id=7,
        thread_id="async-thread",
        user_id="user-async",
        run_id="run-async",
        trace_id="trace-async",
    )

    assert result.success is True
    assert result.answer == "async agent ok"
    assert result.project_id == 7
    assert result.thread_id == "async-thread"
    assert result.effective_thread_id == "project:7:thread:async-thread"
    assert agent.received_thread_id == "project:7:thread:async-thread"
    assert isinstance(agent.received_context, CodeDocRuntimeContext)
    assert agent.received_context.user_id == "user-async"
    assert agent.received_context.project_id == 7
    assert agent.received_context.run_id == "run-async"
    assert agent.received_context.trace_id == "trace-async"
