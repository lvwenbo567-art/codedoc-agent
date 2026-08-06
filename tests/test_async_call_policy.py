from __future__ import annotations

import asyncio

import pytest

from runtime.async_call_policy import AsyncCallController, AsyncCallRetryExhaustedError, AsyncCallTimeoutError, RetryPolicy


@pytest.mark.asyncio
async def test_retries_then_succeeds() -> None:
    attempts = 0
    controller = AsyncCallController[int](max_concurrency=2, timeout_seconds=1,
                                         retry_policy=RetryPolicy(max_attempts=3, base_delay_seconds=0, jitter_seconds=0))
    async def operation() -> int:
        nonlocal attempts
        attempts += 1
        if attempts < 3: raise ConnectionError("temporary")
        return 7
    assert await controller.run(operation, operation_name="unit", retry_if=lambda _: True) == 7
    assert attempts == 3


@pytest.mark.asyncio
async def test_timeout_is_wrapped_after_attempts() -> None:
    controller = AsyncCallController[None](max_concurrency=1, timeout_seconds=0.01,
                                           retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0))
    async def operation() -> None:
        await asyncio.sleep(1)
    with pytest.raises(AsyncCallRetryExhaustedError) as exc:
        await controller.run(operation, operation_name="slow", retry_if=lambda _: True)
    assert isinstance(exc.value.__cause__, AsyncCallTimeoutError)


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency() -> None:
    controller = AsyncCallController[int](max_concurrency=2, timeout_seconds=1,
                                         retry_policy=RetryPolicy(max_attempts=1))
    active = 0
    maximum = 0
    async def operation() -> int:
        nonlocal active, maximum
        active += 1; maximum = max(maximum, active)
        await asyncio.sleep(0.02)
        active -= 1
        return 1
    await asyncio.gather(*(controller.run(operation, operation_name="parallel", retry_if=lambda _: False) for _ in range(5)))
    assert maximum <= 2
