from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

#工具处理器必须是一个可以调用的对象，它可以接收任意参数，也可以返回任意类型。
ToolHandler = Callable[..., Any]


@dataclass(frozen=True)
class ToolSpec:
    """
    一个工具的完整定义。

    name:
        模型调用时使用的工具名。

    description:
        告诉模型什么时候应该使用此工具。

    args_model:
        Pydantic 参数模型。

    handler:
        真正执行工具逻辑的 Python 函数。
    """

    name: str
    description: str
    args_model: type[BaseModel]
    handler: ToolHandler

    def to_openai_tool(self) -> dict:
        """
        转换成 OpenAI-compatible function tool。
        """
        parameters = (
            self.args_model.model_json_schema()
        )

        parameters.pop("title", None)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }


class ToolRegistry:
    """
    工具注册中心，同时也是工具白名单。

    只有注册过的工具才允许执行。
    """

    TOOL_NAME_PATTERN = re.compile(
        r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$"
    )

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        tool: ToolSpec,
    ) -> None:
        if not self.TOOL_NAME_PATTERN.fullmatch(
            tool.name
        ):
            raise ValueError(
                f"非法工具名称：{tool.name}"
            )

        if not tool.description.strip():
            raise ValueError(
                f"工具描述不能为空：{tool.name}"
            )

        if tool.name in self._tools:
            raise ValueError(
                f"工具已经注册：{tool.name}"
            )

        self._tools[tool.name] = tool

    def get(
        self,
        tool_name: str,
    ) -> ToolSpec | None:
        return self._tools.get(tool_name)

    def require(
        self,
        tool_name: str,
    ) -> ToolSpec:
        tool = self.get(tool_name)

        if tool is None:
            raise KeyError(
                f"工具不存在：{tool_name}"
            )

        return tool

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def list_specs(self) -> list[ToolSpec]:
        return [
            self._tools[name]
            for name in self.names()
        ]

    def to_openai_tools(
        self,
    ) -> list[dict]:
        return [
            tool.to_openai_tool()
            for tool in self.list_specs()
        ]

    def __contains__(
        self,
        tool_name: str,
    ) -> bool:
        return tool_name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
