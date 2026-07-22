from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.api_response import (
    error_response,
    http_status_to_error_code,
    success_response,
)
from schemas.api_schema import (
    AskRequest,
    EvalRequest,
    HybridSearchRequest,
    IndexRequest,
    RerankEvalRequest,
    RerankSearchRequest,
    ScanRequest,
    SearchRequest,
    VectorSearchRequest,
)
from config import (
    DEFAULT_BASE_URL,
    DEFAULT_CHAT_BASE_URL,
    DEFAULT_CHAT_MAX_TOKENS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_PROVIDER,
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_CHAT_TIMEOUT_SECONDS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_RETRY_BACKOFF_SECONDS,
    DEFAULT_EMBEDDING_MAX_RETRIES,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    DEFAULT_MODEL_NAME,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_CANDIDATE_TOP_K,
    DEFAULT_RERANK_DEVICE,
    DEFAULT_RERANK_FINAL_TOP_K,
    DEFAULT_RERANK_MAX_LENGTH,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_PROVIDER,
    SUPPORTED_SUFFIXES,
)
from evaluation.eval_service import evaluate_retrieval_from_files
from services.index_service import build_vector_index_from_json
from services.hybrid_search_service import hybrid_search_from_files
from logger import setup_logger
from services.project_service import scan_project
from services.rag_service import ask_from_vector_index
from evaluation.rerank_eval_service import compare_search_methods, load_eval_cases
from pipelines.retrieval_pipeline import retrieve_with_rerank
from repositories.repository import (
    get_chunk_by_id,
    get_file_by_id,
    get_project_by_id,
    list_chunks,
    list_files,
    list_projects,
)
from services.keyword_search_service import search_chunks_from_json
from services.vector_search_service import search_vector_index_from_file
from api.function_calling_router import router as function_calling_router


logger = setup_logger()

app = FastAPI(
    title="CodeDoc Research Agent API",
    description="A FastAPI backend for CodeDoc Research Agent.",
    version="0.1.0",
)

app.include_router(function_calling_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """
    将 HTTPException 统一包装成项目约定的错误响应格式。
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
    将 Pydantic 参数校验错误统一包装成 VALIDATION_ERROR。
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
    健康检查接口，用于确认后端服务是否正常运行。
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
    返回当前项目版本和学习阶段。
    """
    logger.info("调用 /version 接口")

    return success_response(
        data={
            "service": "codedoc-agent",
            "version": "0.1.0",
            "stage": "day28-rerank-stability-eval",
        }
    )


@app.get("/config")
def get_config() -> dict:
    """
    返回当前后端默认配置，但不会返回 API Key 等敏感值。
    """
    logger.info("调用 /config 接口")

    return success_response(
        data={
            "supported_suffixes": sorted(SUPPORTED_SUFFIXES),
            "default_chunk_size": DEFAULT_CHUNK_SIZE,
            "default_chunk_overlap": DEFAULT_CHUNK_OVERLAP,
            "default_model_name": DEFAULT_MODEL_NAME,
            "default_base_url": DEFAULT_BASE_URL,
            "default_chat_provider": DEFAULT_CHAT_PROVIDER,
            "default_chat_model": DEFAULT_CHAT_MODEL,
            "default_chat_base_url": DEFAULT_CHAT_BASE_URL,
            "default_chat_timeout_seconds": DEFAULT_CHAT_TIMEOUT_SECONDS,
            "default_chat_temperature": DEFAULT_CHAT_TEMPERATURE,
            "default_chat_max_tokens": DEFAULT_CHAT_MAX_TOKENS,
            "default_embedding_provider": DEFAULT_EMBEDDING_PROVIDER,
            "default_embedding_model": DEFAULT_EMBEDDING_MODEL,
            "default_embedding_base_url": DEFAULT_EMBEDDING_BASE_URL,
            "default_embedding_timeout_seconds": DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
            "default_embedding_batch_size": DEFAULT_EMBEDDING_BATCH_SIZE,
            "default_embedding_max_retries": DEFAULT_EMBEDDING_MAX_RETRIES,
            "default_embedding_retry_backoff_seconds": DEFAULT_EMBEDDING_RETRY_BACKOFF_SECONDS,
            "default_rerank_provider": DEFAULT_RERANK_PROVIDER,
            "default_rerank_model": DEFAULT_RERANK_MODEL,
            "default_rerank_device": DEFAULT_RERANK_DEVICE,
            "default_rerank_batch_size": DEFAULT_RERANK_BATCH_SIZE,
            "default_rerank_candidate_top_k": DEFAULT_RERANK_CANDIDATE_TOP_K,
            "default_rerank_final_top_k": DEFAULT_RERANK_FINAL_TOP_K,
            "default_rerank_max_length": DEFAULT_RERANK_MAX_LENGTH,
        }
    )


@app.post("/scan")
def scan_project_api(request: ScanRequest) -> dict:
    """
    扫描项目目录，读取文件、切分 chunks，并可选保存到 JSON 和数据库。
    """
    logger.info("调用 /scan 接口，project_path=%s", request.project_path)

    try:
        result = scan_project(
            project_path=request.project_path,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
            save_chunks=request.save_chunks,
            output_path=request.output_path,
            save_to_db=request.save_to_db,
            db_path=request.db_path,
        )

        return success_response(data=result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except NotADirectoryError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/search")
def search_chunks_api(request: SearchRequest) -> dict:
    """
    从 chunks.json 中执行关键词检索，返回 Top-K 相关 chunks。
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
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/eval")
def evaluate_retrieval_api(request: EvalRequest) -> dict:
    """
    根据评测集计算检索效果指标。
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
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/projects")
def list_projects_api(
    db_path: str = "data/codedoc.db",
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """
    分页查询数据库中的项目扫描记录。
    """
    try:
        projects = list_projects(
            limit=limit,
            offset=offset,
            db_path=db_path,
        )

        return success_response(
            data={
                "db_path": db_path,
                "limit": limit,
                "offset": offset,
                "project_count": len(projects),
                "projects": projects,
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/projects/{project_id}")
def get_project_api(
    project_id: int,
    db_path: str = "data/codedoc.db",
) -> dict:
    """
    根据项目 ID 查询单个项目记录。
    """
    project = get_project_by_id(
        project_id=project_id,
        db_path=db_path,
    )

    if project is None:
        raise HTTPException(
            status_code=404,
            detail=f"项目记录不存在：{project_id}",
        )

    return success_response(data=project)


@app.get("/files")
def list_files_api(
    db_path: str = "data/codedoc.db",
    project_id: int | None = None,
    suffix: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """
    分页查询数据库中的文件记录，可按项目和后缀过滤。
    """
    try:
        files = list_files(
            project_id=project_id,
            suffix=suffix,
            limit=limit,
            offset=offset,
            db_path=db_path,
        )

        return success_response(
            data={
                "db_path": db_path,
                "project_id": project_id,
                "suffix": suffix,
                "limit": limit,
                "offset": offset,
                "file_count": len(files),
                "files": files,
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/files/{file_id}")
def get_file_api(
    file_id: int,
    db_path: str = "data/codedoc.db",
) -> dict:
    """
    根据文件 ID 查询单个文件记录。
    """
    file = get_file_by_id(
        file_id=file_id,
        db_path=db_path,
    )

    if file is None:
        raise HTTPException(
            status_code=404,
            detail=f"文件记录不存在：{file_id}",
        )

    return success_response(data=file)


@app.get("/chunks")
def list_chunks_api(
    db_path: str = "data/codedoc.db",
    project_id: int | None = None,
    chunk_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """
    分页查询数据库中的 chunk 记录，可按项目和 chunk 类型过滤。
    """
    try:
        chunks = list_chunks(
            project_id=project_id,
            chunk_type=chunk_type,
            limit=limit,
            offset=offset,
            db_path=db_path,
        )

        return success_response(
            data={
                "db_path": db_path,
                "project_id": project_id,
                "chunk_type": chunk_type,
                "limit": limit,
                "offset": offset,
                "chunk_count": len(chunks),
                "chunks": chunks,
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/chunks/{chunk_db_id}")
def get_chunk_api(
    chunk_db_id: int,
    db_path: str = "data/codedoc.db",
) -> dict:
    """
    根据 chunk 数据库 ID 查询单个 chunk 记录。
    """
    chunk = get_chunk_by_id(
        chunk_db_id=chunk_db_id,
        db_path=db_path,
    )

    if chunk is None:
        raise HTTPException(
            status_code=404,
            detail=f"chunk 不存在：{chunk_db_id}",
        )

    return success_response(data=chunk)


@app.post("/index")
def build_vector_index_api(request: IndexRequest) -> dict:
    """
    从 chunks.json 构建向量索引，支持真实 Embedding、批处理和建库统计。
    """
    embedding_model = request.model_name or request.embedding_model
    mock_dimension = (
        request.dimension
        if request.dimension is not None
        else request.mock_dimension
    )

    logger.info(
        "调用 /index，chunks_path=%s, output_path=%s, embedding_provider=%s, embedding_model=%s",
        request.chunks_path,
        request.output_path,
        request.embedding_provider,
        embedding_model,
    )

    try:
        result = build_vector_index_from_json(
            chunks_path=request.chunks_path,
            output_path=request.output_path,
            embedding_provider=request.embedding_provider,
            embedding_model=embedding_model,
            embedding_base_url=request.embedding_base_url,
            embedding_api_key=request.embedding_api_key,
            timeout_seconds=request.embedding_timeout_seconds,
            mock_dimension=mock_dimension,
            batch_size=request.batch_size,
            incremental=request.incremental,
        )

        return success_response(data=result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))

    except (ConnectionError, RuntimeError) as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/vector_search")
def vector_search_api(request: VectorSearchRequest) -> dict:
    """
    从向量索引中检索与 query 最相似的 Top-K chunks。
    """
    embedding_model = request.model_name or request.embedding_model
    mock_dimension = (
        request.dimension
        if request.dimension is not None
        else request.mock_dimension
    )

    logger.info(
        "调用 /vector_search，index_path=%s, query=%s, top_k=%s",
        request.index_path,
        request.query,
        request.top_k,
    )

    try:
        result = search_vector_index_from_file(
            query=request.query,
            index_path=request.index_path,
            top_k=request.top_k,
            embedding_provider=request.embedding_provider,
            embedding_model=embedding_model,
            embedding_base_url=request.embedding_base_url,
            embedding_api_key=request.embedding_api_key,
            timeout_seconds=request.embedding_timeout_seconds,
            mock_dimension=mock_dimension,
            chunk_type=request.chunk_type,
        )

        return success_response(data=result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))

    except (ConnectionError, RuntimeError) as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/hybrid_search")
def hybrid_search_api(request: HybridSearchRequest) -> dict:
    """
    同时执行关键词检索和向量检索，并按权重融合两个通道的 Top-K 结果。
    """
    embedding_model = request.model_name or request.embedding_model
    mock_dimension = (
        request.dimension
        if request.dimension is not None
        else request.mock_dimension
    )

    logger.info(
        "调用 /hybrid_search，query=%s, chunks_path=%s, index_path=%s",
        request.query,
        request.chunks_path,
        request.index_path,
    )

    try:
        result = hybrid_search_from_files(
            query=request.query,
            chunks_path=request.chunks_path,
            index_path=request.index_path,
            keyword_top_k=request.keyword_top_k,
            vector_top_k=request.vector_top_k,
            final_top_k=request.final_top_k,
            keyword_weight=request.keyword_weight,
            vector_weight=request.vector_weight,
            embedding_provider=request.embedding_provider,
            embedding_model=embedding_model,
            embedding_base_url=request.embedding_base_url,
            embedding_api_key=request.embedding_api_key,
            embedding_timeout_seconds=request.embedding_timeout_seconds,
            mock_dimension=mock_dimension,
            chunk_type=request.chunk_type,
        )

        return success_response(data=result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))

    except (ConnectionError, RuntimeError) as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/rerank_search")
def rerank_search_api(request: RerankSearchRequest) -> dict:
    """
    执行 Hybrid Search + Rerank，两阶段召回精排后返回 Top-K chunks。
    """
    embedding_model = request.model_name or request.embedding_model
    mock_dimension = (
        request.dimension
        if request.dimension is not None
        else request.mock_dimension
    )

    logger.info(
        "调用 /rerank_search，query=%s, chunks_path=%s, index_path=%s",
        request.query,
        request.chunks_path,
        request.index_path,
    )

    try:
        result = retrieve_with_rerank(
            query=request.query,
            chunks_path=request.chunks_path,
            index_path=request.index_path,
            candidate_top_k=request.candidate_top_k,
            final_top_k=request.final_top_k,
            keyword_weight=request.keyword_weight,
            vector_weight=request.vector_weight,
            embedding_provider=request.embedding_provider,
            embedding_model=embedding_model,
            embedding_base_url=request.embedding_base_url,
            embedding_api_key=request.embedding_api_key,
            embedding_timeout_seconds=request.embedding_timeout_seconds,
            mock_dimension=mock_dimension,
            rerank_provider=request.rerank_provider,
            rerank_model=request.rerank_model,
            rerank_device=request.rerank_device,
            rerank_batch_size=request.rerank_batch_size,
            rerank_max_length=request.rerank_max_length,
            rerank_local_files_only=request.rerank_local_files_only,
            chunk_type=request.chunk_type,
            query_strategy=request.query_strategy,
            rewrite_count=request.rewrite_count,
            query_rewrite_provider=request.query_rewrite_provider,
            query_rewrite_model=request.query_rewrite_model,
            query_rewrite_base_url=request.query_rewrite_base_url,
            query_rewrite_api_key=request.query_rewrite_api_key,
            query_rewrite_timeout_seconds=request.query_rewrite_timeout_seconds,
        )

        return success_response(data=result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))

    except (ConnectionError, RuntimeError) as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/rerank_eval")
def rerank_eval_api(request: RerankEvalRequest) -> dict:
    """
    对比 Hybrid Search 和 Hybrid + Rerank 的 Hit@K、MRR 与平均延迟。
    """
    embedding_model = request.model_name or request.embedding_model
    mock_dimension = (
        request.dimension
        if request.dimension is not None
        else request.mock_dimension
    )

    logger.info(
        "调用 /rerank_eval，eval_path=%s, chunks_path=%s, index_path=%s",
        request.eval_path,
        request.chunks_path,
        request.index_path,
    )

    try:
        eval_cases = load_eval_cases(request.eval_path)

        def run_hybrid(query: str) -> dict:
            return hybrid_search_from_files(
                query=query,
                chunks_path=request.chunks_path,
                index_path=request.index_path,
                keyword_top_k=request.candidate_top_k,
                vector_top_k=request.candidate_top_k,
                final_top_k=request.final_top_k,
                keyword_weight=request.keyword_weight,
                vector_weight=request.vector_weight,
                embedding_provider=request.embedding_provider,
                embedding_model=embedding_model,
                embedding_base_url=request.embedding_base_url,
                embedding_api_key=request.embedding_api_key,
                embedding_timeout_seconds=request.embedding_timeout_seconds,
                mock_dimension=mock_dimension,
                chunk_type=request.chunk_type,
            )

        def run_rerank(query: str) -> dict:
            return retrieve_with_rerank(
                query=query,
                chunks_path=request.chunks_path,
                index_path=request.index_path,
                candidate_top_k=request.candidate_top_k,
                final_top_k=request.final_top_k,
                keyword_weight=request.keyword_weight,
                vector_weight=request.vector_weight,
                embedding_provider=request.embedding_provider,
                embedding_model=embedding_model,
                embedding_base_url=request.embedding_base_url,
                embedding_api_key=request.embedding_api_key,
                embedding_timeout_seconds=request.embedding_timeout_seconds,
                mock_dimension=mock_dimension,
                rerank_provider=request.rerank_provider,
                rerank_model=request.rerank_model,
                rerank_device=request.rerank_device,
                rerank_batch_size=request.rerank_batch_size,
                rerank_max_length=request.rerank_max_length,
                rerank_local_files_only=request.rerank_local_files_only,
                chunk_type=request.chunk_type,
                query_strategy="original",
            )

        result = compare_search_methods(
            eval_cases=eval_cases,
            hybrid_search_function=run_hybrid,
            rerank_search_function=run_rerank,
        )

        return success_response(
            data={
                "eval_path": request.eval_path,
                "chunks_path": request.chunks_path,
                "index_path": request.index_path,
                "candidate_top_k": request.candidate_top_k,
                "final_top_k": request.final_top_k,
                **result,
            }
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))

    except (ConnectionError, RuntimeError) as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/ask")
def ask_api(request: AskRequest) -> dict:
    """
    基于向量检索结果执行 RAG 问答。
    """
    embedding_model = request.model_name or request.embedding_model
    mock_dimension = (
        request.dimension
        if request.dimension is not None
        else request.mock_dimension
    )

    logger.info(
        "调用 /ask，query=%s, index_path=%s, top_k=%s",
        request.query,
        request.index_path,
        request.top_k,
    )

    try:
        result = ask_from_vector_index(
            query=request.query,
            retrieval_mode=request.retrieval_mode,
            chunks_path=request.chunks_path,
            index_path=request.index_path,
            top_k=request.top_k,
            candidate_top_k=request.candidate_top_k,
            keyword_weight=request.keyword_weight,
            vector_weight=request.vector_weight,
            embedding_provider=request.embedding_provider,
            embedding_model=embedding_model,
            embedding_base_url=request.embedding_base_url,
            embedding_api_key=request.embedding_api_key,
            embedding_timeout_seconds=request.embedding_timeout_seconds,
            mock_dimension=mock_dimension,
            chat_provider=request.chat_provider,
            chat_model=request.chat_model,
            chat_base_url=request.chat_base_url,
            chat_api_key=request.chat_api_key,
            chat_timeout_seconds=request.chat_timeout_seconds,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            rerank_provider=request.rerank_provider,
            rerank_model=request.rerank_model,
            rerank_device=request.rerank_device,
            rerank_batch_size=request.rerank_batch_size,
            rerank_max_length=request.rerank_max_length,
            rerank_local_files_only=request.rerank_local_files_only,
            chunk_type=request.chunk_type,
            max_context_chars=request.max_context_chars,
            query_strategy=request.query_strategy,
            rewrite_count=request.rewrite_count,
        )

        return success_response(data=result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))

    except (ConnectionError, RuntimeError) as e:
        raise HTTPException(status_code=502, detail=str(e))
