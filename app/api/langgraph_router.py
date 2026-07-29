from __future__ import annotations

from fastapi import APIRouter, HTTPException

from langchain_agent.model_config import LangChainModelConfig
from langgraph_agent.dependencies import build_graph_dependencies
from langgraph_agent.rag_runtime import RAGRuntimeConfig
from langgraph_agent.workflow_service import (
    CodeDocAgenticRAGService,
    CodeDocWorkflowExecutionError,
    CodeDocWorkflowService,
)
from schemas.langgraph_schema import (
    LangGraphWorkflowRequest,
    LangGraphWorkflowResponse,
)


router = APIRouter(
    prefix="/langgraph",
    tags=["langgraph"],
)


def _build_runtime_from_request(
    request: LangGraphWorkflowRequest,
) -> RAGRuntimeConfig:
    return RAGRuntimeConfig(
        project_root=request.project_root,
        chunks_path=request.chunks_path,
        index_path=request.index_path,
        candidate_top_k=request.candidate_top_k,
        final_top_k=request.final_top_k,
        rewrite_count=request.rewrite_count,
        keyword_weight=request.keyword_weight,
        vector_weight=request.vector_weight,
        max_context_chars=request.max_context_chars,
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


def _state_to_response(
    *,
    result: dict,
    request: LangGraphWorkflowRequest,
) -> LangGraphWorkflowResponse:
    return LangGraphWorkflowResponse(
        query=str(result.get("query") or request.query),
        project_id=int(result.get("project_id") or request.project_id),
        query_type=result.get("query_type", "unknown"),
        answer=str(result.get("answer") or ""),
        retrieval_strategy=result.get("retrieval_strategy", "none"),
        symbol_name=result.get("symbol_name"),
        query_decision=dict(result.get("query_decision") or {}),
        evidence_assessment=dict(result.get("evidence_assessment") or {}),
        retrieval_metadata=dict(result.get("retrieval_metadata") or {}),
        evidence=list(result.get("evidence") or []),
        citations=list(result.get("citations") or []),
        answer_quality=dict(result.get("answer_quality") or {}),
        degraded=bool(result.get("degraded")),
        degrade_reasons=list(result.get("degrade_reasons") or []),
        execution_steps=list(result.get("execution_steps") or []),
        evidence_sufficient=bool(result.get("evidence_sufficient")),
        error_message=result.get("error_message"),
    )


def build_workflow_service_for_request(
    request: LangGraphWorkflowRequest,
) -> CodeDocWorkflowService:
    return CodeDocWorkflowService(
        project_root=request.project_root,
        chunks_path=request.chunks_path,
        index_path=request.index_path,
        embedding_provider=request.embedding_provider,
        embedding_model=request.embedding_model,
        embedding_base_url=request.embedding_base_url,
        embedding_api_key=request.embedding_api_key,
        embedding_timeout_seconds=request.embedding_timeout_seconds,
        mock_dimension=request.mock_dimension,
        model_config=LangChainModelConfig.from_env(),
    )


def build_agentic_rag_service_for_request(
    request: LangGraphWorkflowRequest,
) -> CodeDocAgenticRAGService:
    runtime = _build_runtime_from_request(request)
    model_config = LangChainModelConfig.from_env()
    dependencies = build_graph_dependencies(
        runtime=runtime,
        model_config=model_config,
    )

    return CodeDocAgenticRAGService(
        dependencies=dependencies,
        runtime=runtime,
    )


@router.post(
    "/workflow",
    response_model=LangGraphWorkflowResponse,
)
async def run_langgraph_workflow(
    request: LangGraphWorkflowRequest,
) -> LangGraphWorkflowResponse:
    service = build_workflow_service_for_request(request)

    try:
        result = await service.arun(
            query=request.query,
            project_id=request.project_id,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LangGraph Workflow 执行失败：{exc}",
        ) from exc

    return _state_to_response(result=result, request=request)


@router.post(
    "/agentic-rag",
    response_model=LangGraphWorkflowResponse,
)
async def run_agentic_rag(
    request: LangGraphWorkflowRequest,
) -> LangGraphWorkflowResponse:
    try:
        service = build_agentic_rag_service_for_request(request)
        result = await service.arun(
            query=request.query,
            project_id=request.project_id,
            recursion_limit=request.recursion_limit,
        )

        return _state_to_response(result=result, request=request)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CodeDocWorkflowExecutionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Agentic RAG 依赖服务调用失败：{exc}",
        ) from exc
