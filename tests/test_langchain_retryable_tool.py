from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1] / "app"
    )
)


from pydantic import BaseModel

from langchain_agent.tool_adapter import (
    RetryableToolExecutionError,
    build_langchain_tools,
)
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry, ToolSpec


class EmptyArgs(BaseModel):
    pass


def broken_tool() -> dict:
    raise RuntimeError("temporary failure")


def test_retryable_tool_error():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="broken_tool",
            description="Temporary failure.",
            args_model=EmptyArgs,
            handler=broken_tool,
        )
    )
    tools = build_langchain_tools(
        registry=registry,
        executor=ToolExecutor(registry),
    )

    try:
        tools[0].invoke({})
    except RetryableToolExecutionError:
        pass
    else:
        raise AssertionError("应该抛出可重试工具错误")
