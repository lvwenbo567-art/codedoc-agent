from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent.checkpoint_runtime import SQLiteCheckpointRuntime
from langgraph_agent.human_review_service import HumanReviewAgentExecutionError
from langgraph_agent.sse_stream_service import HumanReviewSSEService
from langgraph_agent.thread_identity import validate_public_thread_id
from langgraph_agent.tool_agent_config import ToolAgentRuntimeConfig
from langgraph_agent.tool_agent_dependencies import ToolAgentConfigurationError
from schemas.human_review_api_schema import (
    HITLAgentBaseRequest,
    HITLAgentResponse,
    HITLAgentResumeRequest,
    HITLAgentStartRequest,
)


router = APIRouter(
    prefix="/langgraph/hitl",
    tags=["langgraph-hitl"],
)


def get_checkpoint_runtime(request: Request) -> SQLiteCheckpointRuntime:
    runtime = getattr(request.app.state, "checkpoint_runtime", None)

    if runtime is None:
        raise RuntimeError("Checkpoint Runtime 未初始化")

    return runtime


def _build_runtime_config(body: HITLAgentBaseRequest) -> ToolAgentRuntimeConfig:
    return ToolAgentRuntimeConfig(
        project_root=body.project_root,
        chunks_path=body.chunks_path,
        index_path=body.index_path,
        max_model_calls=body.max_model_calls,
        max_tool_calls=body.max_tool_calls,
        max_identical_tool_calls=body.max_identical_tool_calls,
        max_model_messages=body.max_model_messages,
        trace_content_chars=body.trace_content_chars,
        enable_human_review=body.enable_human_review,
        approval_required_tools=tuple(body.approval_required_tools),
        embedding_provider=body.embedding_provider,
        embedding_model=body.embedding_model,
        embedding_base_url=body.embedding_base_url,
        embedding_api_key=body.embedding_api_key,
        embedding_timeout_seconds=body.embedding_timeout_seconds,
        mock_dimension=body.mock_dimension,
        rerank_provider=body.rerank_provider,
        rerank_model=body.rerank_model,
        rerank_device=body.rerank_device,
        rerank_batch_size=body.rerank_batch_size,
        rerank_max_length=body.rerank_max_length,
        rerank_local_files_only=body.rerank_local_files_only,
    )


async def get_hitl_service(
    *,
    body: HITLAgentBaseRequest,
    request: Request,
):
    validate_public_thread_id(body.thread_id)
    checkpoint_runtime = get_checkpoint_runtime(request)
    runtime_config = _build_runtime_config(body)
    model_config = LangChainModelConfig.from_env()

    return await checkpoint_runtime.get_or_create_hitl_service(
        runtime=runtime_config,
        model_config=model_config,
    )


@router.post("/start", response_model=HITLAgentResponse)
async def start_human_review_agent(
    body: HITLAgentStartRequest,
    request: Request,
) -> HITLAgentResponse:
    try:
        service = await get_hitl_service(body=body, request=request)
        result = await service.start(
            query=body.query,
            project_id=body.project_id,
            thread_id=body.thread_id,
            recursion_limit=body.recursion_limit,
        )

        return HITLAgentResponse.model_validate(result)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (RuntimeError, ValueError, ToolAgentConfigurationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HumanReviewAgentExecutionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/resume", response_model=HITLAgentResponse)
async def resume_human_review_agent(
    body: HITLAgentResumeRequest,
    request: Request,
) -> HITLAgentResponse:
    try:
        service = await get_hitl_service(body=body, request=request)
        result = await service.resume(
            project_id=body.project_id,
            thread_id=body.thread_id,
            decision=body.decision,
            recursion_limit=body.recursion_limit,
        )

        return HITLAgentResponse.model_validate(result)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (RuntimeError, ValueError, ToolAgentConfigurationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HumanReviewAgentExecutionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/stream")
async def stream_human_review_agent(
    body: HITLAgentStartRequest,
    request: Request,
) -> StreamingResponse:
    try:
        service = await get_hitl_service(body=body, request=request)
        sse_service = HumanReviewSSEService(agent_service=service)

        return StreamingResponse(
            sse_service.stream_start(
                query=body.query,
                project_id=body.project_id,
                thread_id=body.thread_id,
                recursion_limit=body.recursion_limit,
                request=request,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except (RuntimeError, ValueError, ToolAgentConfigurationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/resume/stream")
async def stream_resume_human_review_agent(
    body: HITLAgentResumeRequest,
    request: Request,
) -> StreamingResponse:
    try:
        service = await get_hitl_service(body=body, request=request)
        sse_service = HumanReviewSSEService(agent_service=service)

        return StreamingResponse(
            sse_service.stream_resume(
                project_id=body.project_id,
                thread_id=body.thread_id,
                decision=body.decision,
                recursion_limit=body.recursion_limit,
                request=request,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except (RuntimeError, ValueError, ToolAgentConfigurationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
