from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api_response import (
    error_response,
    http_status_to_error_code,
    success_response,
)
from api_schema import EvalRequest, ScanRequest, SearchRequest
from config import (
    DEFAULT_BASE_URL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL_NAME,
    SUPPORTED_SUFFIXES,
)
from eval_service import evaluate_retrieval_from_files
from logger import setup_logger
from project_service import scan_project
from search_service import search_chunks_from_json


logger = setup_logger()

app = FastAPI(
    title="CodeDoc Research Agent API",
    description="A FastAPI backend for CodeDoc Research Agent.",
    version="0.1.0",
)


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """
    统一处理 HTTPException。
    """
    code = http_status_to_error_code(exc.status_code)

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(
            code=code,
            message=str(exc.detail),
        ),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    统一处理请求参数校验错误。
    """
    return JSONResponse(
        status_code=422,
        content=error_response(
            code="VALIDATION_ERROR",
            message="请求参数校验失败",
            details=exc.errors(),
        ),
    )


@app.get("/health")
def health_check() -> dict:
    """
    健康检查接口。

    用于确认 API 服务是否正常运行。
    """
    logger.info("调用 /health 接口")

    return success_response(
        data={
            "status": "ok",
            "service": "codedoc-agent",
        }
    )


@app.get("/version")
def get_version() -> dict:
    """
    获取当前 API 版本信息。
    """
    logger.info("调用 /version 接口")

    return success_response(
        data={
            "service": "codedoc-agent",
            "version": "0.1.0",
            "stage": "day16-eval-response-api",
        }
    )


@app.get("/config")
def get_config() -> dict:
    """
    获取当前项目的基础配置信息。
    """
    logger.info("调用 /config 接口")

    return success_response(
        data={
            "supported_suffixes": sorted(SUPPORTED_SUFFIXES),
            "default_chunk_size": DEFAULT_CHUNK_SIZE,
            "default_chunk_overlap": DEFAULT_CHUNK_OVERLAP,
            "default_model_name": DEFAULT_MODEL_NAME,
            "default_base_url": DEFAULT_BASE_URL,
        }
    )


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

        return success_response(data=result)

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


@app.post("/search")
def search_chunks_api(request: SearchRequest) -> dict:
    """
    从 chunks.json 中检索相关 chunks。
    """
    logger.info(
        "调用 /search 接口，chunks_path=%s, query=%s, top_k=%s",
        request.chunks_path,
        request.query,
        request.top_k,
    )

    try:
        results = search_chunks_from_json(
            input_path=request.chunks_path,
            query=request.query,
            top_k=request.top_k,
        )

        return success_response(
            data={
                "chunks_path": request.chunks_path,
                "query": request.query,
                "top_k": request.top_k,
                "result_count": len(results),
                "results": results,
            }
        )

    except FileNotFoundError as e:
        logger.error("chunks 文件不存在: %s", e)

        raise HTTPException(
            status_code=404,
            detail=str(e),
        )

    except ValueError as e:
        logger.error("检索参数错误: %s", e)

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


@app.post("/eval")
def evaluate_retrieval_api(request: EvalRequest) -> dict:
    """
    执行检索评估。
    """
    logger.info(
        "调用 /eval 接口，chunks_path=%s, eval_path=%s, top_k=%s",
        request.chunks_path,
        request.eval_path,
        request.top_k,
    )

    try:
        result = evaluate_retrieval_from_files(
            chunks_path=request.chunks_path,
            eval_path=request.eval_path,
            top_k=request.top_k,
        )

        return success_response(data=result)

    except FileNotFoundError as e:
        logger.error("评估所需文件不存在: %s", e)

        raise HTTPException(
            status_code=404,
            detail=str(e),
        )

    except ValueError as e:
        logger.error("评估参数错误: %s", e)

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )