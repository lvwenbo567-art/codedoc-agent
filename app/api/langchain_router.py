from __future__ import annotations

from fastapi import APIRouter, HTTPException

from langchain_agent.agent_factory import LangChainAgentConfigurationError
from langchain_agent.agent_schema import LangChainAgentResult
from langchain_agent.agent_service import (
    LangChainAgentExecutionError,
    LangChainAgentService,
)
from langchain_agent.chat_service import LangChainChatService
from langchain_agent.message_builder import ConversationTurn
from langchain_agent.middleware_config import LangChainMiddlewareConfig
from langchain_agent.model_config import LangChainModelConfig
from langchain_agent.output_schema import LangChainChatResult, QueryAnalysisResult
from langchain_agent.structured_output_service import QueryAnalysisService
from schemas.langchain_schema import (
    LangChainAgentRequest,
    LangChainChatRequest,
    QueryAnalysisRequest,
)


router = APIRouter(
    prefix="/langchain",
    tags=["langchain"],
)


def _format_downstream_error(exc: Exception) -> str:
    """
    尽量保留下游模型服务返回的错误细节，方便定位 Ollama/vLLM 502。
    """
    message = str(exc)
    response = getattr(exc, "response", None)

    if response is None:
        return message

    status_code = getattr(response, "status_code", None)

    try:
        body = response.text
    except Exception:
        body = None

    if body:
        return (
            f"{message}; downstream_status={status_code}; "
            f"downstream_body={body}"
        )

    return f"{message}; downstream_status={status_code}"


@router.get("/config")
def get_langchain_config() -> dict:
    """
    返回当前 LangChain 模型配置，不泄露 API Key。
    """
    model_config = LangChainModelConfig.from_env()
    middleware_config = LangChainMiddlewareConfig.from_env()

    return {
        "model": model_config.safe_dict(),
        "middleware": middleware_config.safe_dict(),
    }


@router.post(
    "/chat",
    response_model=LangChainChatResult,
)
def langchain_chat(
    request: LangChainChatRequest,
) -> LangChainChatResult:
    """
    使用 LangChain ChatModel 执行一次普通模型调用。
    """
    config = LangChainModelConfig.from_env()
    history = [
        ConversationTurn(
            role=item.role,
            content=item.content,
        )
        for item in request.history
    ]
    service = LangChainChatService(config=config)

    try:
        return service.ask(
            query=request.query,
            history=history,
        )

    except Exception as exc:
        debug_payload = None

        try:
            debug_payload = service.build_debug_payload(
                query=request.query,
                history=history,
            )
        except Exception as debug_exc:
            debug_payload = {
                "debug_error": str(debug_exc),
            }

        raise HTTPException(
            status_code=502,
            detail=(
                "LangChain 模型调用失败："
                f"{_format_downstream_error(exc)}; "
                f"config={config.safe_dict()}; "
                f"payload={debug_payload}"
            ),
        ) from exc


@router.post(
    "/analyze-query",
    response_model=QueryAnalysisResult,
)
def analyze_query(
    request: QueryAnalysisRequest,
) -> QueryAnalysisResult:
    """
    使用 LangChain Structured Output 分析 Query 类型、推荐工具和查询策略。
    """
    config = LangChainModelConfig.from_env()
    service = QueryAnalysisService(config=config)

    try:
        return service.analyze(request.query)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Query Analysis 失败："
                f"{_format_downstream_error(exc)}"
            ),
        ) from exc


@router.post(
    "/agent",
    response_model=LangChainAgentResult,
)
async def run_langchain_agent(
    request: LangChainAgentRequest,
) -> LangChainAgentResult:
    """
    使用 LangChain create_agent 执行 CodeDoc 工具调用 Agent。
    """
    config = LangChainModelConfig.from_env()
    middleware_config = LangChainMiddlewareConfig.from_env()
    service = LangChainAgentService(
        config=config,
        middleware_config=middleware_config,
        project_root=request.project_root,
        chunks_path=request.chunks_path,
        index_path=request.index_path,
        recursion_limit=request.recursion_limit,
        embedding_provider=request.embedding_provider,
        embedding_model=request.embedding_model,
        embedding_base_url=request.embedding_base_url,
        embedding_api_key=request.embedding_api_key,
        embedding_timeout_seconds=request.embedding_timeout_seconds,
        mock_dimension=request.mock_dimension,
        rerank_provider=request.rerank_provider,
        rerank_model=request.rerank_model,
        rerank_device=request.rerank_device,
        rerank_batch_size=request.rerank_batch_size,
        rerank_max_length=request.rerank_max_length,
        rerank_local_files_only=request.rerank_local_files_only,
    )

    try:
        return await service.arun(
            query=request.query,
            project_id=request.project_id,
            thread_id=request.thread_id,
            user_id=request.user_id,
            project_root=request.project_root,
            chunks_path=request.chunks_path,
            index_path=request.index_path,
            recursion_limit=request.recursion_limit,
            run_id=request.run_id,
            trace_id=request.trace_id,
        )

    except LangChainAgentConfigurationError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=str(exc),
        ) from exc

    except LangChainAgentExecutionError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "LangChain Agent 执行失败："
                f"{_format_downstream_error(exc)}"
            ),
        ) from exc
