from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1] / "app"
    )
)


from langchain_core.messages import HumanMessage

from langchain_agent.model_config import LangChainModelConfig
from langchain_agent.model_factory import create_chat_model


def test_ollama_chat_model_does_not_send_max_completion_tokens():
    config = LangChainModelConfig(
        provider="openai_compatible",
        model_name="qwen2.5:3b",
        base_url="http://localhost:11434/v1",
        api_key="EMPTY",
        max_tokens=300,
    )

    model = create_chat_model(config)
    payload = model._get_request_payload(
        [HumanMessage(content="你好")]
    )

    assert "max_completion_tokens" not in payload


def test_non_ollama_chat_model_keeps_token_limit():
    config = LangChainModelConfig(
        provider="openai_compatible",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        base_url="http://localhost:8001/v1",
        api_key="EMPTY",
        max_tokens=300,
    )

    model = create_chat_model(config)
    payload = model._get_request_payload(
        [HumanMessage(content="你好")]
    )

    assert payload["max_completion_tokens"] == 300

