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
)

from function_calling.client import (
    ModelFunctionCall,
    ModelToolCall,
    ModelTurn,
)
from function_calling.loop import (
    ManualFunctionCallingLoop,
)
from tools.executor import ToolExecutor
from tools.registry import (
    ToolRegistry,
    ToolSpec,
)


class EchoArgs(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    text: str


def echo(
    text: str,
) -> dict:
    return {
        "echo": text,
    }


class FakeClient:
    def __init__(self):
        self.call_count = 0
        self.received_messages = []

    def complete(
        self,
        messages,
        tools,
    ):
        self.call_count += 1

        self.received_messages.append(
            list(messages)
        )

        if self.call_count == 1:
            return ModelTurn(
                tool_calls=[
                    ModelToolCall(
                        id="call-1",
                        function=(
                            ModelFunctionCall(
                                name="echo",
                                arguments=(
                                    '{"text": '
                                    '"hello"}'
                                ),
                            )
                        ),
                    )
                ]
            )

        return ModelTurn(
            content=(
                "工具返回了 hello。"
            )
        )


def build_echo_registry():
    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="echo",
            description="返回输入文本。",
            args_model=EchoArgs,
            handler=echo,
        )
    )

    return registry


def test_function_calling_loop():
    registry = (
        build_echo_registry()
    )

    executor = ToolExecutor(
        registry
    )

    client = FakeClient()

    loop = ManualFunctionCallingLoop(
        client=client,
        registry=registry,
        executor=executor,
        max_steps=4,
    )

    result = loop.run(
        "请调用 echo"
    )

    assert (
        result.stop_reason
        == "final_answer"
    )

    assert result.model_call_count == 2
    assert result.tool_call_count == 1

    assert (
        result.tool_traces[0]
        .tool_name
        == "echo"
    )

    assert (
        result.tool_traces[0]
        .result.success
        is True
    )

    second_request_messages = (
        client.received_messages[1]
    )

    tool_messages = [
        message
        for message
        in second_request_messages
        if message["role"] == "tool"
    ]

    assert len(tool_messages) == 1

    assert (
        tool_messages[0][
            "tool_call_id"
        ]
        == "call-1"
    )


class MultiToolClient:
    def __init__(self):
        self.call_count = 0

    def complete(
        self,
        messages,
        tools,
    ):
        self.call_count += 1

        if self.call_count == 1:
            return ModelTurn(
                tool_calls=[
                    ModelToolCall(
                        id="call-1",
                        function=ModelFunctionCall(
                            name="echo",
                            arguments='{"text": "one"}',
                        ),
                    ),
                    ModelToolCall(
                        id="call-2",
                        function=ModelFunctionCall(
                            name="echo",
                            arguments='{"text": "two"}',
                        ),
                    ),
                ]
            )

        return ModelTurn(
            content="两个工具都执行了。"
        )


def test_loop_supports_multiple_tool_calls():
    registry = build_echo_registry()
    loop = ManualFunctionCallingLoop(
        client=MultiToolClient(),
        registry=registry,
        executor=ToolExecutor(registry),
        max_steps=4,
    )

    result = loop.run("调用两次 echo")

    assert result.stop_reason == "final_answer"
    assert result.tool_call_count == 2
    assert [
        trace.tool_call_id
        for trace in result.tool_traces
    ] == ["call-1", "call-2"]


class EndlessToolClient:
    def complete(
        self,
        messages,
        tools,
    ):
        if not tools:
            return ModelTurn(
                content=(
                    "达到限制后的最终回答。"
                )
            )

        return ModelTurn(
            tool_calls=[
                ModelToolCall(
                    id="call-loop",
                    function=(
                        ModelFunctionCall(
                            name="echo",
                            arguments=(
                                '{"text": "loop"}'
                            ),
                        )
                    ),
                )
            ]
        )


def test_loop_stops_at_max_steps():
    registry = (
        build_echo_registry()
    )

    loop = ManualFunctionCallingLoop(
        client=EndlessToolClient(),
        registry=registry,
        executor=ToolExecutor(
            registry
        ),
        max_steps=2,
    )

    result = loop.run(
        "一直调用工具"
    )

    assert (
        result.stop_reason
        == "max_steps"
    )

    assert result.tool_call_count == 2

    assert result.answer == (
        "达到限制后的最终回答。"
    )
