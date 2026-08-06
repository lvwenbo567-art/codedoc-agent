from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx#httpx 是 Python 的 HTTP 客户端库。


@dataclass(frozen=True)
class AsyncHTTPConfig:
    connect_timeout_seconds: float = 5.0#控制“建立连接”最多等待多久
    read_timeout_seconds: float = 60.0#控制“连接建立后，等待服务器返回响应内容”最多多久。
    write_timeout_seconds: float = 30.0#控制“把请求体发送给服务端”最多多久。
    pool_timeout_seconds: float = 5.0#控制“等待连接池提供一个可用连接”最多多久。
    max_connections: int = 50#整个共享 HTTP Client 最多可同时持有 50 个连接。
    max_keepalive_connections: int = 20#请求结束后，并非所有连接都立即关闭。其中最多保留 20 个空闲连接，以便下一次继续复用。
    keepalive_expiry_seconds: float = 30.0#空闲连接最多保留 30 秒。


    # CodeDoc 默认调用本机 Ollama/Qdrant；避免系统代理接管 localhost 请求。
    trust_env: bool = False


class AsyncHTTPRuntime:
    """在 FastAPI 生命周期内维护一个共享连接池。
    它不是 HTTP Client 本身；
它内部持有一个 HTTP Client。
    """

    def __init__(self, *, config: AsyncHTTPConfig) -> None:
        self.config = config
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    @property#@property 的作用是让你可以这样访问：runtime.client
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("AsyncHTTPRuntime 尚未启动")
        return self._client

    async def start(self) -> None:
        async with self._lock:
            if self._client is not None:
                return
            self._client = httpx.AsyncClient(
                trust_env=self.config.trust_env,
                timeout=httpx.Timeout(
                    connect=self.config.connect_timeout_seconds,
                    read=self.config.read_timeout_seconds,
                    write=self.config.write_timeout_seconds,
                    pool=self.config.pool_timeout_seconds,
                ),
                limits=httpx.Limits(
                    max_connections=self.config.max_connections,
                    max_keepalive_connections=self.config.max_keepalive_connections,
                    keepalive_expiry=self.config.keepalive_expiry_seconds,
                ),
                follow_redirects=False,#关闭重定向
            )

    async def close(self) -> None:
        async with self._lock:
            client, self._client = self._client, None
            if client is not None:
                await client.aclose()
