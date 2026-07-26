from pathlib import Path
import json
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1] / "app"
    )
)


from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_agent.agent_service import LangChainAgentService
from langchain_agent.model_config import LangChainModelConfig


class FakeAgent:
    def invoke(self, input, config):
        tool_result = {
            "success": True,
            "tool_name": "search_code",
            "data": {
                "result_count": 1,
            },
            "error_code": None,
            "error_message": None,
            "duration_ms": 12.5,
        }

        return {
            "messages": [
                HumanMessage(content="RerankClient 在哪里？"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_code",
                            "args": {
                                "query": "RerankClient",
                            },
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content=json.dumps(
                        tool_result,
                        ensure_ascii=False,
                    ),
                    tool_call_id="call-1",
                    name="search_code",
                ),
                AIMessage(
                    content=(
                        "RerankClient 定义在 "
                        "app/clients/rerank_client.py。"
                    )
                ),
            ]
        }


def test_langchain_agent_service(tmp_path):
    chunks_path = tmp_path / "chunks.json"
    index_path = tmp_path / "index.json"
    chunks_path.write_text("[]", encoding="utf-8")
    index_path.write_text("{}", encoding="utf-8")

    service = LangChainAgentService(
        config=LangChainModelConfig(
            provider="openai_compatible",
            model_name="fake-model",
            base_url="http://localhost/v1",
        ),
        project_root=str(tmp_path),
        chunks_path=str(chunks_path),
        index_path=str(index_path),
        agent=FakeAgent(),
    )

    result = service.run("RerankClient 在哪里？")

    assert result.answer == "RerankClient 定义在 app/clients/rerank_client.py。"
    assert result.tool_call_count == 1
    assert result.tool_traces[0].tool_name == "search_code"
    assert result.tool_traces[0].arguments["query"] == "RerankClient"
    assert result.tool_traces[0].success is True
    assert result.tool_traces[0].duration_ms == 12.5
