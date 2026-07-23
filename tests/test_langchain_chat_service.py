from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1] / "app"
    )
)


from langchain_core.messages import AIMessage

from langchain_agent.chat_service import LangChainChatService, extract_message_text
from langchain_agent.model_config import LangChainModelConfig


class FakeChatModel:
    def __init__(self):
        self.received_messages = None

    def invoke(self, messages):
        self.received_messages = messages

        return AIMessage(
            content="这是测试回答。",
            response_metadata={
                "model_name": "fake",
            },
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        )


def test_chat_service_invokes_model():
    model = FakeChatModel()

    config = LangChainModelConfig(
        provider="openai_compatible",
        model_name="fake-model",
        base_url="http://localhost/v1",
    )

    service = LangChainChatService(
        config=config,
        model=model,
    )

    result = service.ask(query="解释项目结构")

    assert result.answer == "这是测试回答。"
    assert result.message_count == 2
    assert result.usage_metadata["total_tokens"] == 15
    assert model.received_messages is not None


def test_mock_chat_service():
    config = LangChainModelConfig(provider="mock")
    service = LangChainChatService(config=config)

    result = service.ask(query="测试问题")

    assert "测试问题" in result.answer
    assert result.provider == "mock"


def test_extract_message_text_from_content_blocks():
    message = AIMessage(
        content=[
            {
                "type": "text",
                "text": "第一段",
            },
            {
                "type": "text",
                "text": "第二段",
            },
        ]
    )

    assert extract_message_text(message) == "第一段\n第二段"

