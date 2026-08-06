from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from runtime.async_http_gateway import AsyncHTTPGateway


@dataclass(frozen=True)
class AsyncRerankConfig:
    base_url: str
    model_name: str
    api_key: str | None = None

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/rerank"


class AsyncRerankClient:
    """兼容常见 /rerank 协议的异步 HTTP Rerank 客户端。"""

    def __init__(self, *, config: AsyncRerankConfig, gateway: AsyncHTTPGateway) -> None:
        self.config = config
        self.gateway = gateway

    async def rerank(self, *, query: str, documents: list[str], top_n: int | None = None) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query 不能为空")
        if not documents:
            return []
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        response = await self.gateway.post_json(
            url=self.config.endpoint,
            payload={"model": self.config.model_name, "query": query, "documents": documents,
                     **({"top_n": top_n} if top_n is not None else {})},
            headers=headers,
            operation_name="rerank",
        )
        results = response.get("results")
        if not isinstance(results, list):
            raise ValueError("Rerank 响应缺少 results")
        return [dict(item) for item in results if isinstance(item, dict)]
