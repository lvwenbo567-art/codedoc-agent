from __future__ import annotations

from typing import Any

import httpx

from runtime.async_call_policy import AsyncCallController
from runtime.async_http_runtime import AsyncHTTPRuntime


RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
'''
408 Request Timeout
服务端认为请求等待时间过长，可能是临时情况。
409 Conflict
某些服务在资源临时冲突时会返回，当前项目保守地允许有限重试。
425 Too Early
服务端暂时不愿处理请求，可稍后再试。
429 Too Many Requests
限流。不能马上连续重试，需要通过退避等待后再试。
500 Internal Server Error
502 Bad Gateway
503 Service Unavailable
504 Gateway Timeout
这些常见于：
Ollama 模型加载中
vLLM 后端暂时不可用
反向代理与模型服务通信失败
Qdrant 或网关临时异常
它们通常是临时故障，因此可有限重试。
'''

class AsyncHTTPGateway:
    """共享 HTTP Client 上的 JSON 请求网关。
    发送 JSON POST
状态码检查
JSON 格式检查
重试规则
    """

    def __init__(self, *, runtime: AsyncHTTPRuntime, controller: AsyncCallController[dict[str, Any]]) -> None:
        self.runtime = runtime
        self.controller = controller

    @staticmethod
    def _should_retry(exception: Exception) -> bool:
        if isinstance(exception, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                                  httpx.WriteTimeout, httpx.PoolTimeout, httpx.RemoteProtocolError)):
            return True
        return isinstance(exception, httpx.HTTPStatusError) and exception.response.status_code in RETRYABLE_STATUS_CODES

    async def post_json(self, *, url: str, payload: dict[str, Any], headers: dict[str, str] | None, operation_name: str) -> dict[str, Any]:
        async def operation() -> dict[str, Any]:
            response = await self.runtime.client.post(url, json=payload, headers=headers)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                # 保留受控长度的下游响应，供上层识别 Ollama 的 NaN 类错误。
                response_text = response.text.strip()
                if not response_text:
                    raise

                if len(response_text) > 500:
                    response_text = response_text[:500] + "..."

                raise httpx.HTTPStatusError(
                    (
                        f"{operation_name} 服务返回状态码："
                        f"{response.status_code}；响应内容：{response_text}"
                    ),
                    request=exc.request,
                    response=exc.response,
                ) from exc
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError(f"{operation_name} 响应不是 JSON Object")
            return result

        return await self.controller.run(operation, operation_name=operation_name, retry_if=self._should_retry)
