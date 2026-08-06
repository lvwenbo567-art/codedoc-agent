from __future__ import annotations

from contextlib import asynccontextmanager
import os

'''这个用来写 FastAPI 的 lifespan。
FastAPI 支持：
app = FastAPI(lifespan=app_lifespan)
app_lifespan 需要是一个异步上下文管理器。'''



from fastapi import FastAPI

from langgraph_agent.checkpoint_config import CheckpointConfig
from langgraph_agent.checkpoint_runtime import SQLiteCheckpointRuntime
from clients.async_embedding_client import AsyncEmbeddingClient, AsyncEmbeddingConfig
from clients.async_chat_client import AsyncChatClient, AsyncChatConfig
from clients.async_rerank_client import AsyncRerankClient, AsyncRerankConfig
from clients.embedding_client import EmbeddingConfig
from ingestion.async_ingestion_pipeline import build_default_async_ingestion_pipeline
from ingestion.ingestion_job_manager import IngestionJobManager
from ingestion.ingestion_job_repository import IngestionJobRepository
from runtime.async_call_policy import AsyncCallController, RetryPolicy
from runtime.async_http_gateway import AsyncHTTPGateway
from runtime.async_http_runtime import AsyncHTTPConfig, AsyncHTTPRuntime
from vectorstores.config import VectorStoreConfig
from vectorstores.factory import create_async_vector_store
from memory.memory_repository import MemoryRepository


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """
    FastAPI 生命周期：启动时打开 SQLite Checkpoint Runtime，关闭时释放连接。
    """
    memory_repository = MemoryRepository(
        database_path=os.getenv(
            "CODEDOC_MEMORY_DB_PATH",
            "data/codedoc_memory.sqlite",
        )
    )
    await memory_repository.start()
    checkpoint_config = CheckpointConfig.from_env()
    checkpoint_runtime = SQLiteCheckpointRuntime(
        config=checkpoint_config,
        memory_repository=memory_repository,
    )

    await checkpoint_runtime.start()
    app.state.checkpoint_runtime = checkpoint_runtime
    app.state.memory_repository = memory_repository
    '''
    它把 runtime 存到 FastAPI 应用状态中。
这样后续 Router 可以通过：
request.app.state.checkpoint_runtime
拿到同一个 Runtime。
也就是：
FastAPI 启动时创建一个 Runtime
所有请求共享这个 Runtime
    '''
    http_runtime = AsyncHTTPRuntime(config=AsyncHTTPConfig())
    await http_runtime.start()

    job_repository = IngestionJobRepository(
        database_path=os.getenv("CODEDOC_INGESTION_JOB_DB_PATH", "data/ingestion_jobs.sqlite")
    )
    await job_repository.start()

    vector_store = create_async_vector_store(VectorStoreConfig.from_env())
    embedding_config = EmbeddingConfig()
    gateway = AsyncHTTPGateway(
        runtime=http_runtime,
        controller=AsyncCallController(
            max_concurrency=int(os.getenv("CODEDOC_ASYNC_MODEL_MAX_CONCURRENCY", "2")),
            timeout_seconds=float(os.getenv("CODEDOC_ASYNC_CALL_TIMEOUT_SECONDS", "60")),
            retry_policy=RetryPolicy(
                max_attempts=int(os.getenv("CODEDOC_ASYNC_MAX_ATTEMPTS", "3")),
                base_delay_seconds=float(os.getenv("CODEDOC_ASYNC_RETRY_BASE_DELAY_SECONDS", "0.5")),
                max_delay_seconds=float(os.getenv("CODEDOC_ASYNC_RETRY_MAX_DELAY_SECONDS", "8")),
                jitter_seconds=float(os.getenv("CODEDOC_ASYNC_RETRY_JITTER_SECONDS", "0.2")),
            ),
        ),
    )
    async_embedding_client = None
    if embedding_config.provider != "mock":
        async_embedding_client = AsyncEmbeddingClient(
            config=AsyncEmbeddingConfig(
                provider=embedding_config.provider,
                base_url=embedding_config.base_url,
                model_name=embedding_config.model_name,
                api_key=embedding_config.api_key or None,
                batch_size=int(os.getenv("CODEDOC_EMBEDDING_BATCH_SIZE", "32")),
            ),
            gateway=gateway,
        )
    ingestion_runner = build_default_async_ingestion_pipeline(
        vector_store=vector_store,
        async_embedding_client=async_embedding_client,
        embedding_config=embedding_config,
    )
    ingestion_manager = IngestionJobManager(
        repository=job_repository,
        runner=ingestion_runner,
        max_running_jobs=int(os.getenv("CODEDOC_INGESTION_MAX_RUNNING_JOBS", "2")),
    )
    await ingestion_manager.start()
    app.state.async_http_runtime = http_runtime
    app.state.async_http_gateway = gateway
    app.state.async_chat_client = AsyncChatClient(
        config=AsyncChatConfig(
            base_url=os.getenv("CODEDOC_CHAT_BASE_URL", "http://localhost:11434/v1"),
            model_name=os.getenv("CODEDOC_CHAT_MODEL", "mock-chat-model"),
            api_key=os.getenv("CODEDOC_CHAT_API_KEY", "") or None,
        ),
        gateway=gateway,
    )
    app.state.async_rerank_client = AsyncRerankClient(
        config=AsyncRerankConfig(
            base_url=os.getenv("CODEDOC_RERANK_BASE_URL", "http://localhost:8001/v1"),
            model_name=os.getenv("CODEDOC_RERANK_MODEL", "mock-reranker"),
            api_key=os.getenv("CODEDOC_RERANK_API_KEY", "") or None,
        ),
        gateway=gateway,
    )
    app.state.ingestion_job_manager = ingestion_manager

    try:
        yield
    finally:
        await ingestion_manager.close()
        await vector_store.close()
        await job_repository.close()
        await http_runtime.close()
        await checkpoint_runtime.close()
        await memory_repository.close()
