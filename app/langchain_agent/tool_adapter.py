from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
'''

这两个类型来自 LangChain。

BaseTool：LangChain 所有工具的基础类型；
StructuredTool：支持结构化参数、Pydantic 参数模型的具体工具类型。

可以理解为：

BaseTool
   ↑
StructuredTool

你最终创建的是 StructuredTool，但函数返回类型写成更通用的：

BaseTool

因此将来即使混入其他 LangChain 工具类型，也可以放进同一个列表。

'''
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry, ToolSpec


RETRYABLE_TOOL_ERROR_CODES = {
    "TOOL_EXECUTION_ERROR",
    "MODEL_SERVICE_ERROR",
    "TIMEOUT",
    "SERVICE_UNAVAILABLE",
}


class RetryableToolExecutionError(RuntimeError):
    """
    工具发生临时错误时抛出，供重试逻辑识别。
    """


def _is_retryable_tool_result(error_code: str | None) -> bool:
    return error_code in RETRYABLE_TOOL_ERROR_CODES


def _build_tool_handler(
    *,
    tool_spec: ToolSpec,
    executor: ToolExecutor,
) -> Callable[..., str]:
    """
    为每个自研 ToolSpec 创建一个 LangChain 可以执行的 Python Handler。

    LangChain 负责：
    - 根据 args_schema 解析模型参数
    - 调用这个 Handler

    自研 ToolExecutor 继续负责：
    - 白名单检查
    - 二次 Pydantic 校验
    - 统一 ToolResult
    - 异常转换
    """

    def run_tool(**kwargs: Any) -> str:#**kwargs 会把所有“具名参数”收集成一个字典。
        result = executor.execute(
            tool_name=tool_spec.name,
            arguments=kwargs,
        )

        if (
            result.success is False
            and _is_retryable_tool_result(
                result.error_code
            )
        ):
            raise RetryableToolExecutionError(
                result.error_message
                or result.error_code
                or "工具执行失败"
            )

        return result.model_dump_json()

    run_tool.__name__ = f"run_{tool_spec.name}"
    run_tool.__doc__ = tool_spec.description

    return run_tool


def convert_tool_spec(
    *,
    tool_spec: ToolSpec,
    executor: ToolExecutor,
) -> BaseTool:
    """
    将一个自研 ToolSpec 适配成 LangChain StructuredTool。
    """
    handler = _build_tool_handler(
        tool_spec=tool_spec,
        executor=executor,
    )

    return StructuredTool.from_function(#根据一个 Python 函数构造 LangChain 工具
        func=handler,
        name=tool_spec.name,
        description=tool_spec.description,
        args_schema=tool_spec.args_model,
        infer_schema=False,#不要让 LangChain根据 handler 的函数签名自动推断参数结构
        return_direct=False,#工具执行结果返回给 Agent，让 Agent继续分析和组织最终回答，而不是把工具结果直接作为最终答案结束运行。
    )


def build_langchain_tools(
    *,
    registry: ToolRegistry,
    executor: ToolExecutor,
) -> list[BaseTool]:
    """
    将 Day30 的全部自研工具统一适配为 LangChain StructuredTool。
    """
    return [
        convert_tool_spec(
            tool_spec=tool_spec,
            executor=executor,
        )
        for tool_spec in registry.list_specs()
    ]
