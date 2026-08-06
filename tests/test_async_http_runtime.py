from __future__ import annotations

import pytest

from runtime.async_http_runtime import AsyncHTTPConfig, AsyncHTTPRuntime


@pytest.mark.asyncio
async def test_http_runtime_reuses_single_client_and_closes() -> None:
    runtime = AsyncHTTPRuntime(config=AsyncHTTPConfig())
    assert runtime.config.trust_env is False
    with pytest.raises(RuntimeError):
        _ = runtime.client
    await runtime.start()
    client = runtime.client
    await runtime.start()
    assert runtime.client is client
    await runtime.close()
    assert client.is_closed
