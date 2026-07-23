from __future__ import annotations

import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from langchain_agent.message_builder import ConversationTurn, build_chat_messages
from langchain_agent.model_config import LangChainModelConfig
from langchain_agent.model_factory import create_chat_model
from langchain_agent.output_schema import LangChainChatResult


def extract_message_text(message: AIMessage) -> str:
    """
    从 AIMessage 中提取文本，兼容字符串 content 和标准内容块。
    """
    content = message.content

    if isinstance(content, str):
        text = content.strip()

        if text:
            return text

        # Ollama 的部分 thinking 模型可能把内容放在非标准 reasoning 字段中。
        reasoning = message.additional_kwargs.get(
            "reasoning"
        )

        if reasoning:
            return str(reasoning).strip()

        return ""

    if not isinstance(content, list):
        return str(content).strip()
    '''
    模型可能返回：

content = [
    {
        "type": "text",
        "text": "项目入口文件是 main.py。"
    },
    {
        "type": "text",
        "text": "它会创建 FastAPI 应用。"
    },
]
    '''
    text_parts: list[str] = []

    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
            continue

        if not isinstance(block, dict):
            continue

        block_type = block.get("type")

        if block_type == "text":
            text = block.get("text", "")

            if text:
                text_parts.append(str(text))

    return "\n".join(text_parts).strip()


class LangChainChatService:
    """
    Day31 独立模型调用服务。

    今天只验证：
    Message 构建 -> model.invoke() -> AIMessage 读取。
    不包含 Tool 或 Agent Loop。
    """

    def __init__(
        self,
        *,
        config: LangChainModelConfig,
        model: BaseChatModel | Any | None = None,
    ) -> None:
        self.config = config
        self._model = model

    def _get_model(self) -> BaseChatModel:
        """
        懒加载真实 ChatModel，避免 mock 和单元测试阶段触发外部连接。
        """
        if self._model is None:
            self._model = create_chat_model(self.config)

        return self._model

    def ask(
        self,
        *,
        query: str,
        history: list[ConversationTurn] | None = None,
    ) -> LangChainChatResult:
        """
        执行一次 LangChain 普通聊天调用。
        """
        start_time = time.perf_counter()

        messages = build_chat_messages(
            query=query,
            history=history,
        )

        if self.config.provider == "mock" and self._model is None:
            answer = (
                "LangChain Mock 模式已收到问题："
                f"{query.strip()}"
            )

            return LangChainChatResult(
                query=query.strip(),
                answer=answer,
                provider=self.config.provider,
                model_name=self.config.model_name,
                message_count=len(messages),
                duration_ms=self._duration_ms(start_time),
            )

        model = self._get_model()
        response = model.invoke(messages)

        if not isinstance(response, AIMessage):
            raise TypeError("LangChain ChatModel 应返回 AIMessage")

        answer = extract_message_text(response)

        if not answer:
            raise ValueError("模型返回了空回答")

        return LangChainChatResult(
            query=query.strip(),
            answer=answer,
            provider=self.config.provider,
            model_name=self.config.model_name,
            message_count=len(messages),
            response_metadata=dict(response.response_metadata or {}),
            usage_metadata=dict(response.usage_metadata or {}),
            duration_ms=self._duration_ms(start_time),
        )

    def build_debug_payload(
        self,
        *,
        query: str,
        history: list[ConversationTurn] | None = None,
    ) -> dict:
        """
        构造 LangChain 即将发送给模型服务的请求 payload，用于排查兼容问题。
        """
        messages = build_chat_messages(
            query=query,
            history=history,
        )
        model = self._get_model()
        get_payload = getattr(
            model,
            "_get_request_payload",
            None,
        )

        if get_payload is None:
            return {
                "debug_error": "当前 ChatModel 不支持 _get_request_payload",
                "message_count": len(messages),
            }

        return get_payload(messages)

    @staticmethod
    def _duration_ms(start_time: float) -> float:
        """
        计算本次模型调用耗时，单位毫秒。
        """
        return round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )
