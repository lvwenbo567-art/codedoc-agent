from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictRequestModel(BaseModel):
    """
    LangChain API 请求基类。

    extra="forbid" 用来拒绝未声明字段，避免调用方传入拼错或多余参数时静默通过。
    """

    model_config = ConfigDict(
        extra="forbid",
    )


class HistoryMessage(StrictRequestModel):
    """
    /langchain/chat 的历史消息结构。
    """

    role: Literal[
        "user",
        "assistant",
    ]

    content: str = Field(
        min_length=1,
        max_length=10000,
    )


class LangChainChatRequest(StrictRequestModel):
    """
    /langchain/chat 请求体。
    """

    query: str = Field(
        min_length=1,
        max_length=3000,
    )

    history: list[HistoryMessage] = Field(
        default_factory=list,
        max_length=20,
    )


class QueryAnalysisRequest(StrictRequestModel):
    """
    /langchain/analyze-query 请求体。
    """

    query: str = Field(
        min_length=1,
        max_length=1000,
    )

