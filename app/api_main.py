from fastapi import FastAPI

from config import (
    DEFAULT_BASE_URL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL_NAME,
    SUPPORTED_SUFFIXES,
)

from logger import setup_logger

logger=setup_logger()

app=FastAPI(
    title="CodeDoc Research Agent API",
    description="A FastAPI backend for CodeDoc Research Agent.",
    version="0.1.0",
)

@app.get("/health")
def health_check() -> dict:
    """
    健康检查接口。

    用于确认 API 服务是否正常运行。
    """
    logger.info("调用 /health 接口")

    return {
        "status": "ok",
        "service": "codedoc-agent",
    }

@app.get("/version")
def get_version() -> dict:
    """
    获取当前 API 版本信息。
    """
    logger.info("调用 /version 接口")

    return {
        "service": "codedoc-agent",
        "version": "0.1.0",
        "stage": "day13-fastapi-entry",
    }

@app.get("/config")
def get_config() -> dict:
    """
    获取当前项目的基础配置信息。
    """
    logger.info("调用 /config 接口")

    return {
        "supported_suffixes": sorted(SUPPORTED_SUFFIXES),
        "default_chunk_size": DEFAULT_CHUNK_SIZE,
        "default_chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "default_model_name": DEFAULT_MODEL_NAME,
        "default_base_url": DEFAULT_BASE_URL,
    }