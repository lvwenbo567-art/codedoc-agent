from __future__ import annotations

from contextlib import asynccontextmanager

'''这个用来写 FastAPI 的 lifespan。
FastAPI 支持：
app = FastAPI(lifespan=app_lifespan)
app_lifespan 需要是一个异步上下文管理器。'''



from fastapi import FastAPI

from langgraph_agent.checkpoint_config import CheckpointConfig
from langgraph_agent.checkpoint_runtime import SQLiteCheckpointRuntime


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """
    FastAPI 生命周期：启动时打开 SQLite Checkpoint Runtime，关闭时释放连接。
    """
    checkpoint_config = CheckpointConfig.from_env()
    checkpoint_runtime = SQLiteCheckpointRuntime(
        config=checkpoint_config
    )

    await checkpoint_runtime.start()
    app.state.checkpoint_runtime = checkpoint_runtime
    '''
    它把 runtime 存到 FastAPI 应用状态中。
这样后续 Router 可以通过：
request.app.state.checkpoint_runtime
拿到同一个 Runtime。
也就是：
FastAPI 启动时创建一个 Runtime
所有请求共享这个 Runtime
    '''
    try:
        yield
    finally:
        await checkpoint_runtime.close()
