from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx

from config import (
    DEFAULT_CHAT_API_KEY,
    DEFAULT_CHAT_BASE_URL,
    DEFAULT_CHAT_MAX_TOKENS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_PROVIDER,
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_CHAT_TIMEOUT_SECONDS,
)


@dataclass(frozen=True)
class ModelFunctionCall:
    """
    模型返回的 function 调用信息。
    """

    name: str
    arguments: str


@dataclass(frozen=True)
class ModelToolCall:
    """
    模型返回的一次工具调用。
    """
    '''
    ModelToolCall(
    id="call_123",
    type="function",
    function=ModelFunctionCall(
        name="search_code",
        arguments='{"query": "EmbeddingClient"}',
    ),
)
    '''
    id: str
    function: ModelFunctionCall
    type: str = "function"


@dataclass(frozen=True)
class ModelTurn:
    """
    模型一轮输出。

    content:
        普通最终回答。

    tool_calls:
        模型希望应用端执行的工具调用列表。
    """

    content: str | None = None
    tool_calls: list[ModelToolCall] | None = None

    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class FunctionCallingClient:
    """
    Function Calling 模型客户端。

    支持 mock 和 OpenAI-compatible /chat/completions。
    """

    def __init__(
        self,
        provider: str = DEFAULT_CHAT_PROVIDER,
        model_name: str = DEFAULT_CHAT_MODEL,
        base_url: str = DEFAULT_CHAT_BASE_URL,
        api_key: str = DEFAULT_CHAT_API_KEY,
        timeout_seconds: float = DEFAULT_CHAT_TIMEOUT_SECONDS,
        temperature: float = DEFAULT_CHAT_TEMPERATURE,
        max_tokens: int = DEFAULT_CHAT_MAX_TOKENS,
        http_client: httpx.Client | None = None,
    ) -> None:
        if provider not in {
            "mock",
            "openai_compatible",
        }:
            raise ValueError(
                f"不支持的 Function Calling Provider：{provider}"
            )

        if not model_name.strip():
            raise ValueError("model_name 不能为空")

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds 必须大于 0"
            )

        self.provider = provider
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.http_client = http_client

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict],
    ) -> ModelTurn:
        """
        调用模型完成一轮 Function Calling。
        """
        if self.provider == "mock":
            return self._complete_mock(
                messages=messages,
                tools=tools,
            )

        return self._complete_openai_compatible(
            messages=messages,
            tools=tools,
        )

    def _complete_mock(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict],
    ) -> ModelTurn:
        """
        Mock 模型：根据 query 关键词选择工具，方便本地测试。
        """
        tool_messages = [
            message
            for message in messages
            if message.get("role") == "tool"
        ]

        if tool_messages:
            return ModelTurn(
                content=(
                    "我已经根据工具返回结果完成分析。"
                )
            )

        user_message = next(
            (
                message.get("content", "")
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )

        if not tools:
            return ModelTurn(
                content=(
                    "当前没有可用工具，无法执行工具检索。"
                )
            )

        tool_name = self._choose_mock_tool(
            query=user_message,
        )

        arguments: dict[str, Any]

        if tool_name == "get_project_structure":
            arguments = {
                "max_depth": 4,
                "max_entries": 300,
                "include_files": True,
                "include_hidden": False,
            }
        else:
            arguments = {
                "query": user_message,
                "top_k": 5,
                "candidate_top_k": 20,
                "query_strategy": "multi_query",
            }

        import json

        return ModelTurn(
            tool_calls=[
                ModelToolCall(
                    id=f"call_{uuid4().hex[:12]}",
                    function=ModelFunctionCall(
                        name=tool_name,
                        arguments=json.dumps(
                            arguments,
                            ensure_ascii=False,
                        ),
                    ),
                )
            ]
        )

    @staticmethod
    def _choose_mock_tool(
        query: str,
    ) -> str:
        lowered = query.lower()

        if any(
            keyword in lowered
            for keyword in [
                "目录",
                "结构",
                "模块",
                "文件树",
                "project structure",
            ]
        ):
            return "get_project_structure"

        if any(
            keyword in lowered
            for keyword in [
                "readme",
                "文档",
                "启动",
                "说明",
                "怎么用",
            ]
        ):
            return "search_documents"

        return "search_code"

    def _complete_openai_compatible(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict],
    ) -> ModelTurn:
        """
        调用 OpenAI-compatible Chat Completions。
        """
        if not self.base_url.strip():
            raise ValueError(
                "openai_compatible Provider 必须配置 base_url"
            )

        url = f"{self.base_url.rstrip('/')}/chat/completions"

        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = (
                f"Bearer {self.api_key}"
            )

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        if (
            "localhost:11434" in self.base_url
            or "127.0.0.1:11434" in self.base_url
        ):
            payload["think"] = False

        owns_client = self.http_client is None
        client = self.http_client or httpx.Client(
            timeout=self.timeout_seconds,
            trust_env=False,
        )

        try:
            response = client.post(
                url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        except httpx.TimeoutException as exc:
            raise TimeoutError(
                "Function Calling 模型请求超时"
            ) from exc

        except httpx.RequestError as exc:
            raise ConnectionError(
                f"无法连接 Function Calling 模型服务：{exc}"
            ) from exc

        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            response_text = exc.response.text.strip()
            if len(response_text) > 500:
                response_text = response_text[:500] + "..."

            detail = (
                "Function Calling 模型服务返回状态码："
                f"{status_code}"
            )
            if response_text:
                detail = (
                    f"{detail}；响应内容：{response_text}"
                )

            raise RuntimeError(detail) from exc

        finally:
            if owns_client:
                client.close()

        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "Function Calling 模型返回结构不符合预期"
            ) from exc

        return self._parse_message(message)
    '''
    
    把模型返回的原始消息字典，转换成项目内部统一使用的 ModelTurn 对象。

模型原始返回通常是这样的：

message = {
    "content": None,
    "tool_calls": [
        {
            "id": "call_001",
            "type": "function",
            "function": {
                "name": "search_code",
                "arguments": '{"query": "EmbeddingClient"}',
            },
        }
    ],
}

转换后变成：

ModelTurn(
    content=None,
    tool_calls=[
        ModelToolCall(
            id="call_001",
            type="function",
            function=ModelFunctionCall(
                name="search_code",
                arguments='{"query": "EmbeddingClient"}',
            ),
        )
    ],
)
    '''
    @staticmethod
    def _parse_message(
        message: dict[str, Any],
    ) -> ModelTurn:
        raw_tool_calls = message.get("tool_calls") or []

        tool_calls: list[ModelToolCall] = []

        if raw_tool_calls:
            if not isinstance(raw_tool_calls, list):
                raise ValueError(
                    "tool_calls 必须是列表"
                )

            for raw_call in raw_tool_calls:
                function = raw_call.get(
                    "function",
                    {},
                )

                name = function.get("name")
                arguments = function.get(
                    "arguments",
                    "{}",
                )

                if not isinstance(name, str):
                    raise ValueError(
                        "tool_call.function.name 必须是字符串"
                    )

                if not isinstance(arguments, str):
                    raise ValueError(
                        "tool_call.function.arguments 必须是字符串"
                    )

                tool_calls.append(
                    ModelToolCall(
                        id=str(
                            raw_call.get(
                                "id",
                                f"call_{uuid4().hex[:12]}",
                            )
                        ),
                        type=str(
                            raw_call.get(
                                "type",
                                "function",
                            )
                        ),
                        function=ModelFunctionCall(
                            name=name,
                            arguments=arguments,
                        ),
                    )
                )

        content = message.get("content")

        if content is not None and not isinstance(
            content,
            str,
        ):
            raise ValueError(
                "message.content 必须是字符串或 None"
            )

        return ModelTurn(
            content=content,
            tool_calls=tool_calls,
        )
