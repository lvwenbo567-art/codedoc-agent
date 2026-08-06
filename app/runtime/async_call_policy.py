from __future__ import annotations

import asyncio#Semaphore、超时、异步 sleep、取消控制。
'''
Semaphore	限制同时运行的请求数
timeout	限制一次调用最长等待时间
sleep	重试前等待
CancelledError	处理用户取消任务
'''
import random
from collections.abc import Awaitable, Callable#表示“可以被 await 的对象”，例如协程调用结果。
from dataclasses import dataclass
from typing import Generic, TypeVar#让控制器既能处理 dict 响应，也能处理 list、字符串或任何类型的结果。


T = TypeVar("T")
RetryPredicate = Callable[[Exception], bool]
'''
ConnectError → True
ReadTimeout → True
429 → True
502 → True

400 → False
401 → False
403 → False
404 → False
'''

class AsyncCallError(RuntimeError):
    """异步外部调用的基础异常。
    继承 RuntimeError 的原因是：
    不是参数格式错误，不是用户输入验证错误，而是程序运行时调用外部服务出的问题。
    """


class AsyncCallTimeoutError(AsyncCallError):
    """异步外部调用超时。"""


class AsyncCallRetryExhaustedError(AsyncCallError):
    """重试次数耗尽。"""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0
    jitter_seconds: float = 0.2

    def validate(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts 必须大于 0")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("重试等待时间不能小于 0")
        if self.jitter_seconds < 0:
            raise ValueError("jitter_seconds 不能小于 0")

    def get_delay(self, attempt: int) -> float:
        delay = min(
            self.base_delay_seconds * 2 ** max(0, attempt - 1),
            self.max_delay_seconds,
        )
        if self.jitter_seconds:
            delay += random.uniform(0, self.jitter_seconds)
        return delay


class AsyncCallController(Generic[T]):
    """统一处理并发上限、超时、有限重试与退避。
    控制器返回什么结果，由调用者决定。
    """

    def __init__(self, *, max_concurrency: int, timeout_seconds: float, retry_policy: RetryPolicy) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency 必须大于 0")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0")
        retry_policy.validate()
        self.timeout_seconds = timeout_seconds
        self.retry_policy = retry_policy
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def run(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        operation_name: str,
        retry_if: RetryPredicate,
    ) -> T:
        last_error: Exception | None = None
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            try:
                async with self._semaphore:#获取并发许可
                    async with asyncio.timeout(self.timeout_seconds):#只有拿到 Semaphore 后才开始计算 Timeout。
                        return await operation()
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                last_error = AsyncCallTimeoutError(
                    f"{operation_name} 执行超时：{self.timeout_seconds}s"
                )
                retryable = True
            except Exception as exc:
                last_error = exc
                retryable = retry_if(exc)

            if not retryable or attempt >= self.retry_policy.max_attempts:
                break
            await asyncio.sleep(self.retry_policy.get_delay(attempt))

        raise AsyncCallRetryExhaustedError(
            f"{operation_name} 失败，已尝试 {self.retry_policy.max_attempts} 次：{last_error}"
        ) from last_error
