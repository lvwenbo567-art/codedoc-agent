from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx

from config import (
    DEFAULT_CHAT_API_KEY,
    DEFAULT_CHAT_BASE_URL,
    DEFAULT_CHAT_MAX_TOKENS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_PROVIDER,
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_CHAT_TIMEOUT_SECONDS,
)


@dataclass(frozen=True)
class ChatConfig:
    """
    Chat 模型调用配置。
    """

    provider: str = DEFAULT_CHAT_PROVIDER
    model_name: str = DEFAULT_CHAT_MODEL
    base_url: str = DEFAULT_CHAT_BASE_URL
    api_key: str = DEFAULT_CHAT_API_KEY
    timeout_seconds: float = DEFAULT_CHAT_TIMEOUT_SECONDS
    temperature: float = DEFAULT_CHAT_TEMPERATURE
    max_tokens: int = DEFAULT_CHAT_MAX_TOKENS

    def validate(self) -> None:
        """
        在发送请求前校验 Chat 配置是否合法。
        """
        if self.provider not in {"mock", "openai_compatible"}:
            raise ValueError(f"不支持的 Chat Provider：{self.provider}")

        if not self.model_name.strip():
            raise ValueError("model_name 不能为空")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0")

        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature 必须在 0 到 2 之间")

        if self.max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0")

        if self.provider == "openai_compatible" and not self.base_url.strip():
            raise ValueError("openai_compatible Provider 必须配置 base_url")


class ChatClient:
    """
    Chat 模型客户端。

    支持：
    1. mock
    2. openai_compatible
    """

    def __init__(
        self,
        config: ChatConfig,
        http_client: Optional[httpx.Client] = None,
    ):
        """
        初始化 Chat 客户端，并允许测试时注入 httpx.Client。
        """
        config.validate()

        self.config = config
        self.http_client = http_client

    def generate(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """
        根据 messages 生成模型回答。
        """
        self._validate_messages(messages)

        if self.config.provider == "mock":
            return self._generate_mock(messages)

        return self._generate_openai_compatible(messages)

    def _validate_messages(
        self,
        messages: List[Dict[str, str]],
    ) -> None:
        """
        校验 Chat messages 是否符合 role/content 基本结构。
        """
        if not messages:
            raise ValueError("messages 不能为空")

        valid_roles = {"system", "user", "assistant"}

        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if role not in valid_roles:
                raise ValueError(f"不支持的 message role：{role}")

            if not isinstance(content, str):
                raise ValueError("message content 必须是字符串")

            if not content.strip():
                raise ValueError("message content 不能为空")

    def _generate_mock(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """
        生成确定性的 Mock 回答，便于本地测试完整 RAG 链路。
        """
        user_message = messages[-1]["content"]

        if "[Source 1]" not in user_message:
            return (
                "当前检索结果不足，无法根据项目内容可靠回答该问题。"
            )

        return (
            "根据当前检索到的项目内容，可以从最相关的代码或文档片段中"
            f"分析该问题。当前使用的模型是 {self.config.model_name}。[Source 1]"
        )

    def _generate_openai_compatible(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """
        调用 OpenAI-compatible /chat/completions 接口生成回答。
        """
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"

        headers = {
            "Content-Type": "application/json",
        }

        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }

        if "localhost:11434" in self.config.base_url or "127.0.0.1:11434" in self.config.base_url:
            payload["think"] = False

        owns_client = self.http_client is None
        client = self.http_client or httpx.Client(
            timeout=self.config.timeout_seconds,
            trust_env=False,
        )

        try:
            response = client.post(
                url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        except httpx.TimeoutException as exc:
            raise TimeoutError("Chat 模型请求超时") from exc

        except httpx.RequestError as exc:
            raise ConnectionError(
                f"无法连接 Chat 模型服务：{exc}"
            ) from exc

        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            response_text = exc.response.text.strip()
            if len(response_text) > 500:
                response_text = response_text[:500] + "..."

            detail = f"Chat 模型服务返回状态码：{status_code}"
            if response_text:
                detail = f"{detail}；响应内容：{response_text}"

            raise RuntimeError(
                detail
            ) from exc

        finally:
            if owns_client:
                client.close()

        try:
            answer = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Chat 模型返回结构不符合预期") from exc

        if not isinstance(answer, str):
            raise ValueError("Chat 模型返回内容不是字符串")

        if not answer.strip():
            raise ValueError("Chat 模型返回了空回答")

        return answer.strip()


def generate_chat_response(
    messages: List[Dict[str, str]],
    provider: str = DEFAULT_CHAT_PROVIDER,
    model_name: str = DEFAULT_CHAT_MODEL,
    base_url: str = DEFAULT_CHAT_BASE_URL,
    api_key: str = DEFAULT_CHAT_API_KEY,
    timeout_seconds: float = DEFAULT_CHAT_TIMEOUT_SECONDS,
    temperature: float = DEFAULT_CHAT_TEMPERATURE,
    max_tokens: int = DEFAULT_CHAT_MAX_TOKENS,
) -> str:
    """
    服务层统一调用入口：创建 ChatClient 并生成回答。
    """
    config = ChatConfig(
        provider=provider,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    client = ChatClient(config=config)

    return client.generate(messages)
