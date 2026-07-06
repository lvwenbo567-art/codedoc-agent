from fastapi import FastAPI, HTTPException

from api_schema import ScanRequest
from config import (
    DEFAULT_BASE_URL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL_NAME,
    SUPPORTED_SUFFIXES,
)
from logger import setup_logger
from project_service import scan_project


logger = setup_logger()

app = FastAPI(
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
        "stage": "day14-scan-api",
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


@app.post("/scan")
def scan_project_api(request: ScanRequest) -> dict:
    """
    扫描项目目录并构建 chunks。
    """
    logger.info("调用 /scan 接口，project_path=%s", request.project_path)

    try:
        result = scan_project(
            project_path=request.project_path,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
            save_chunks=request.save_chunks,
            output_path=request.output_path,
        )

        return {
            "success": True,
            "data": result,
        }

    except FileNotFoundError as e:
        logger.error("项目路径不存在: %s", e)

        raise HTTPException(
            status_code=404,
            detail=str(e),
        )

    except NotADirectoryError as e:
        logger.error("项目路径不是目录: %s", e)

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except ValueError as e:
        logger.error("参数错误: %s", e)

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )