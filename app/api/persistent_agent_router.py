from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent.checkpoint_inspection import CheckpointInspectionService
from langgraph_agent.checkpoint_runtime import SQLiteCheckpointRuntime
from langgraph_agent.persistent_tool_agent_service import (
    PersistentToolAgentExecutionError,
)
from langgraph_agent.thread_identity import (
    build_effective_thread_id,
    validate_public_thread_id,
)
from langgraph_agent.tool_agent_config import ToolAgentRuntimeConfig
from langgraph_agent.tool_agent_dependencies import ToolAgentConfigurationError
from schemas.persistent_agent_schema import (
    DeleteThreadResponse,
    PersistentAgentRequest,
    PersistentAgentResponse,
    ThreadHistoryResponse,
    ThreadStateResponse,
)


router = APIRouter(
    prefix="/langgraph",
    tags=["langgraph-persistence"],
)


def get_checkpoint_runtime(
    request: Request,
) -> SQLiteCheckpointRuntime:
    """
    从 FastAPI app.state 中获取 SQLiteCheckpointRuntime。
    """
    runtime = getattr(
        request.app.state,
        "checkpoint_runtime",
        None,
    )

    if runtime is None:
        raise RuntimeError(
            "Checkpoint Runtime 未初始化"
        )

    return runtime


def _build_runtime_config(
    body: PersistentAgentRequest,
) -> ToolAgentRuntimeConfig:
    """
    从 API 请求构建 Tool Agent 运行配置。
    """
    return ToolAgentRuntimeConfig(
        project_root=body.project_root,
        chunks_path=body.chunks_path,
        index_path=body.index_path,
        max_model_calls=body.max_model_calls,
        max_tool_calls=body.max_tool_calls,
        max_identical_tool_calls=body.max_identical_tool_calls,
        max_model_messages=body.max_model_messages,
        trace_content_chars=body.trace_content_chars,
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


@router.post(
    "/persistent-agent",
    response_model=PersistentAgentResponse,
)
async def run_persistent_agent(
    body: PersistentAgentRequest,
    request: Request,
) -> PersistentAgentResponse:
    """
    执行带 SQLite Checkpoint 的 Tool Agent。
    """
    try:
        validate_public_thread_id(body.thread_id)
        checkpoint_runtime = get_checkpoint_runtime(request)
        agent_runtime = _build_runtime_config(body)
        model_config = LangChainModelConfig.from_env()

        service = await checkpoint_runtime.get_or_create_service(
            runtime=agent_runtime,
            model_config=model_config,
        )
        result = await service.arun(
            query=body.query,
            project_id=body.project_id,
            thread_id=body.thread_id,
            recursion_limit=body.recursion_limit,
        )

        return PersistentAgentResponse.model_validate(result)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (
        ValueError,
        ToolAgentConfigurationError,
    ) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PersistentToolAgentExecutionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/threads/{thread_id}/state",
    response_model=ThreadStateResponse,
)
async def get_thread_state(
    thread_id: str,
    request: Request,
    project_id: int = Query(ge=1),
) -> ThreadStateResponse:
    """
    查询某个 thread 的最新 State。
    """
    runtime = get_checkpoint_runtime(request)
    service = CheckpointInspectionService(
        checkpointer=runtime.checkpointer
    )
    result = await service.get_latest_state(
        project_id=project_id,
        thread_id=thread_id,
    )

    return ThreadStateResponse.model_validate(result)


@router.get(
    "/threads/{thread_id}/history",
    response_model=ThreadHistoryResponse,
)
async def get_thread_history(
    thread_id: str,
    request: Request,
    project_id: int = Query(ge=1),
    limit: int = Query(default=20, ge=1, le=100),
) -> ThreadHistoryResponse:
    """
    查询某个 thread 的 Checkpoint History。
    """
    runtime = get_checkpoint_runtime(request)
    service = CheckpointInspectionService(
        checkpointer=runtime.checkpointer
    )
    checkpoints = await service.list_history(
        project_id=project_id,
        thread_id=thread_id,
        limit=limit,
    )

    return ThreadHistoryResponse(
        project_id=project_id,
        thread_id=thread_id,
        count=len(checkpoints),
        checkpoints=checkpoints,
    )


@router.delete(
    "/threads/{thread_id}",
    response_model=DeleteThreadResponse,
)
async def delete_thread(
    thread_id: str,
    request: Request,
    project_id: int = Query(ge=1),
) -> DeleteThreadResponse:
    """
    删除某个 thread 的所有 Checkpoint。
    """
    runtime = get_checkpoint_runtime(request)
    service = CheckpointInspectionService(
        checkpointer=runtime.checkpointer
    )
    effective_thread_id = build_effective_thread_id(
        project_id=project_id,
        thread_id=thread_id,
    )
    thread_lock = runtime.get_thread_lock(effective_thread_id)

    async with thread_lock:
        result = await service.delete_thread(
            project_id=project_id,
            thread_id=thread_id,
        )

    return DeleteThreadResponse.model_validate(result)
