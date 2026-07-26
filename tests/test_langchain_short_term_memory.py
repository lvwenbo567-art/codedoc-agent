from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1] / "app"
    )
)


from langchain_core.messages import AIMessage

from langchain_agent.agent_service import LangChainAgentService
from langchain_agent.model_config import LangChainModelConfig


class FakeMemoryAgent:
    def __init__(self) -> None:
        self.messages_by_thread: dict[str, list] = {}

    def invoke(self, input, config, context=None):
        thread_id = config["configurable"]["thread_id"]
        history = self.messages_by_thread.setdefault(
            thread_id,
            [],
        )
        new_messages = input["messages"]
        query = new_messages[-1].content

        if "RerankClient" in query:
            answer = "Remembered: the rerank client is RerankClient."
        elif "it" in query.lower() and any(
            "RerankClient" in message.content
            for message in history
            if hasattr(message, "content")
        ):
            answer = (
                "It refers to RerankClient, defined in "
                "app/clients/rerank_client.py."
            )
        else:
            answer = "Insufficient evidence to resolve the reference."

        history.extend(new_messages)
        history.append(AIMessage(content=answer))

        return {
            "messages": list(history),
        }


def build_service(tmp_path, agent):
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


def test_same_thread_keeps_short_term_memory(tmp_path):
    agent = FakeMemoryAgent()
    service = build_service(tmp_path, agent)

    first = service.run(
        "The project uses a rerank client named RerankClient.",
        project_id=1,
        thread_id="chat-001",
    )
    second = service.run(
        "Which file is it in?",
        project_id=1,
        thread_id="chat-001",
    )

    assert first.effective_thread_id == "project:1:thread:chat-001"
    assert "RerankClient" in second.answer
    assert second.history_message_count > 0
