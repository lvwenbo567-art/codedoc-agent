from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.api_response import success_response
from function_calling.client import FunctionCallingClient
from function_calling.loop import ManualFunctionCallingLoop
from schemas.function_calling_schema import FunctionCallRequest
from tools.code_doc_tools import build_code_doc_tool_registry
from tools.executor import ToolExecutor


router = APIRouter(
    prefix="/agent",
    tags=["agent"],
)


@router.post("/function-call")
def function_call_api(
    request: FunctionCallRequest,
) -> dict:
    """
    手写 Function Calling 入口。

    由模型决定调用哪个只读工具，应用端负责执行工具并把结果返回模型。
    """
    try:
        mock_dimension = (
            request.dimension
            if request.dimension is not None
            else request.mock_dimension
        )

        registry = build_code_doc_tool_registry(
            project_root=request.project_root,
            chunks_path=request.chunks_path,
            index_path=request.index_path,
            embedding_provider=request.embedding_provider,
            embedding_model=request.embedding_model,
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
            query_rewrite_provider=request.provider,
            query_rewrite_model=request.model_name,
            query_rewrite_base_url=request.base_url,
            query_rewrite_api_key=request.api_key,
            query_rewrite_timeout_seconds=request.timeout_seconds,
        )
        executor = ToolExecutor(
            registry=registry,
        )
        client = FunctionCallingClient(
            provider=request.provider,
            model_name=request.model_name,
            base_url=request.base_url,
            api_key=request.api_key,
            timeout_seconds=request.timeout_seconds,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        loop = ManualFunctionCallingLoop(
            client=client,
            registry=registry,
            executor=executor,
            max_steps=request.max_steps,
        )

        result = loop.run(
            query=request.query,
        )

        return success_response(
            data=result.to_dict()
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=str(exc),
        )

    except (ConnectionError, RuntimeError) as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        )
