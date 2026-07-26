from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1] / "app"
    )
)


from pydantic import BaseModel, ConfigDict

from langchain_agent.tool_adapter import build_langchain_tools
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry, ToolSpec


class EchoArgs(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    text: str


def echo(text: str) -> dict:
    return {
        "echo": text,
    }


def test_convert_custom_tool_to_langchain():
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

    tools = build_langchain_tools(
        registry=registry,
        executor=executor,
    )

    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "echo"
    assert tool.args_schema is EchoArgs

    output = tool.invoke({"text": "hello"})

    assert '"success":true' in output
    assert '"echo":"hello"' in output

