from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1]
        / "app"
    )
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from tools.errors import ToolBusinessError
from tools.executor import ToolExecutor
from tools.registry import (
    ToolRegistry,
    ToolSpec,
)


class EchoArgs(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    text: str = Field(min_length=1)


def echo(
    text: str,
) -> dict:
    return {
        "echo": text,
    }


def test_registry_generates_openai_schema():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            description="返回输入文本。",
            args_model=EchoArgs,
            handler=echo,
        )
    )

    tools = registry.to_openai_tools()

    assert registry.names() == ["echo"]
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "echo"
    assert "parameters" in tools[0]["function"]


def test_registry_rejects_duplicate_tool():
    registry = ToolRegistry()
    spec = ToolSpec(
        name="echo",
        description="返回输入文本。",
        args_model=EchoArgs,
        handler=echo,
    )

    registry.register(spec)

    try:
        registry.register(spec)
    except ValueError as exc:
        assert "工具已经注册" in str(exc)
    else:
        raise AssertionError("重复注册应该失败")


def test_executor_success_with_json_arguments():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            description="返回输入文本。",
            args_model=EchoArgs,
            handler=echo,
        )
    )
    executor = ToolExecutor(registry)

    result = executor.execute(
        tool_name="echo",
        arguments='{"text": "hello"}',
    )

    assert result.success is True
    assert result.data == {"echo": "hello"}


def test_executor_rejects_unknown_tool():
    executor = ToolExecutor(ToolRegistry())

    result = executor.execute(
        tool_name="delete_project",
        arguments={},
    )

    assert result.success is False
    assert result.error_code == "TOOL_NOT_FOUND"


def test_executor_rejects_invalid_json():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            description="返回输入文本。",
            args_model=EchoArgs,
            handler=echo,
        )
    )
    executor = ToolExecutor(registry)

    result = executor.execute(
        tool_name="echo",
        arguments="{bad-json",
    )

    assert result.success is False
    assert result.error_code == "INVALID_JSON"


def test_executor_rejects_invalid_arguments():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            description="返回输入文本。",
            args_model=EchoArgs,
            handler=echo,
        )
    )
    executor = ToolExecutor(registry)

    result = executor.execute(
        tool_name="echo",
        arguments={"text": ""},
    )

    assert result.success is False
    assert result.error_code == "INVALID_ARGUMENTS"


def test_executor_handles_business_error():
    def fail() -> dict:
        raise ToolBusinessError(
            error_code="EXPECTED_ERROR",
            message="业务失败",
        )

    class EmptyArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="fail",
            description="失败工具。",
            args_model=EmptyArgs,
            handler=fail,
        )
    )
    executor = ToolExecutor(registry)

    result = executor.execute(
        tool_name="fail",
        arguments={},
    )

    assert result.success is False
    assert result.error_code == "EXPECTED_ERROR"


def test_executor_handles_unexpected_error():
    def explode() -> dict:
        raise RuntimeError("boom")

    class EmptyArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="explode",
            description="异常工具。",
            args_model=EmptyArgs,
            handler=explode,
        )
    )
    executor = ToolExecutor(registry)

    result = executor.execute(
        tool_name="explode",
        arguments={},
    )

    assert result.success is False
    assert result.error_code == "TOOL_EXECUTION_ERROR"
