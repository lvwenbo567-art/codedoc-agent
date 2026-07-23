from __future__ import annotations

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from langchain_agent.model_config import LangChainModelConfig


class LangChainModelConfigurationError(ValueError):
    """
    LangChain 模型配置错误。
    """


def _is_ollama_base_url(base_url: str) -> bool:
    """
    判断当前 OpenAI-compatible 服务是否是 Ollama。

    LangChain/OpenAI SDK 1.x 会把 max_tokens 转成 max_completion_tokens，
    但 Ollama 的 OpenAI-compatible 接口对该字段兼容不好，可能直接返回 502。
    """
    normalized_base_url = base_url.lower()

    return (
        "localhost:11434" in normalized_base_url
        or "127.0.0.1:11434" in normalized_base_url
        or "ollama" in normalized_base_url
    )


def create_chat_model(config: LangChainModelConfig) -> BaseChatModel:
    """
    根据配置创建真实 LangChain ChatModel。

    mock 模式由 Service 自己处理，不在这里伪造 BaseChatModel。
    """
    if config.provider == "mock":
        raise LangChainModelConfigurationError(
            "mock 模式不创建真实 ChatModel"
        )

    if config.provider == "openai_compatible":
        chat_model_kwargs = {
            "model": config.model_name,
            "base_url": config.base_url,
            "api_key": config.api_key.get_secret_value(),
            "temperature": config.temperature,
            "timeout": config.timeout_seconds,
            "max_retries": config.max_retries,
            "http_client": httpx.Client(
                timeout=config.timeout_seconds,
                trust_env=False,
            ),
        }

        if not _is_ollama_base_url(config.base_url):
            chat_model_kwargs["max_tokens"] = config.max_tokens

        return ChatOpenAI(
            **chat_model_kwargs,
        )

    raise LangChainModelConfigurationError(
        f"不支持的 LangChain provider：{config.provider}"
    )
