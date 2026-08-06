from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from runtime.async_http_gateway import AsyncHTTPGateway


@dataclass(frozen=True)
class AsyncChatConfig:
    base_url: str
    model_name: str
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 800

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


class AsyncChatClient:
    """OpenAI-compatible Chat Completions 的异步适配器。"""

    def __init__(self, *, config: AsyncChatConfig, gateway: AsyncHTTPGateway) -> None:
        self.config = config
        self.gateway = gateway

    async def complete(self, *, messages: list[dict[str, str]], temperature: float | None = None,
                       max_tokens: int | None = None) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        response = await self.gateway.post_json(
            url=self.config.endpoint,
            payload={"model": self.config.model_name, "messages": messages, "stream": False,
                     "temperature": self.config.temperature if temperature is None else temperature,
                     "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens},
            headers=headers,
            operation_name="chat",
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Chat 响应缺少 choices")
        return response
