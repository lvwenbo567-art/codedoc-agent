from __future__ import annotations

from fastapi import APIRouter, HTTPException

from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent.tool_agent_config import ToolAgentRuntimeConfig
from langgraph_agent.tool_agent_dependencies import (
    ToolAgentConfigurationError,
    build_tool_agent_dependencies,
)
from langgraph_agent.tool_agent_service import (
    CodeDocToolAgentExecutionError,
    CodeDocToolAgentService,
)
from schemas.tool_agent_schema import ToolAgentRequest, ToolAgentResponse


router = APIRouter(
    prefix="/langgraph",
    tags=["langgraph"],
)


def build_tool_agent_service(
    request: ToolAgentRequest,
) -> CodeDocToolAgentService:
    runtime = ToolAgentRuntimeConfig(
        project_root=request.project_root,
        chunks_path=request.chunks_path,
        index_path=request.index_path,
        max_model_calls=request.max_model_calls,
        max_tool_calls=request.max_tool_calls,
        max_identical_tool_calls=request.max_identical_tool_calls,
        max_model_messages=request.max_model_messages,
        trace_content_chars=request.trace_content_chars,
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
    model_config = LangChainModelConfig.from_env()
    dependencies = build_tool_agent_dependencies(
        runtime=runtime,
        model_config=model_config,
    )

    return CodeDocToolAgentService(
        dependencies=dependencies,
        runtime=runtime,
    )


@router.post(
    "/tool-agent",
    response_model=ToolAgentResponse,
)
async def run_tool_agent(
    request: ToolAgentRequest,
) -> ToolAgentResponse:
    try:
        service = build_tool_agent_service(request)
        result = await service.arun(
            query=request.query,
            project_id=request.project_id,
            recursion_limit=request.recursion_limit,
        )

        return ToolAgentResponse.model_validate(result)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, ToolAgentConfigurationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CodeDocToolAgentExecutionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Tool Agent 依赖服务调用失败：{exc}",
        ) from exc
