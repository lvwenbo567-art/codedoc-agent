from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


QueryType = Literal[
    "code",
    "document",
    "structure",
    "mixed",
    "unknown",
]

RecommendedTool = Literal[
    "search_code",
    "search_documents",
    "get_project_structure",
    "multiple",
    "none",
]

RecommendedQueryStrategy = Literal[
    "original",
    "rewrite",
    "multi_query",
]

AnalysisMode = Literal[
    "mock_rules",
    "structured_output",
    "rule_fallback",
]


class StrictOutputModel(BaseModel):
    """
    结构化输出基类：不允许模型返回 Schema 之外的字段。
    """

    model_config = ConfigDict(
        extra="forbid",
    )


class QueryAnalysis(StrictOutputModel):
    """
    供后续 Retrieval Router 和 LangGraph 条件路由使用的 Query 分析结果。
    """

    query_type: QueryType = Field(
        description="用户问题的主要类型。"
    )
    recommended_tool: RecommendedTool = Field(
        description="建议调用的 CodeDoc 工具。"
    )
    recommended_query_strategy: RecommendedQueryStrategy = Field(
        description="建议使用的检索查询策略。"
    )
    needs_rewrite: bool = Field(
        description="是否建议先改写用户查询。"
    )
    protected_terms: list[str] = Field(
        default_factory=list,
        description="改写时必须保留的函数名、类名、文件名、常量或 API 路径。",
    )
    classification_reason: str = Field(
        min_length=1,
        max_length=300,
        description="简短说明分类依据，不输出详细思维过程。",
    )


class QueryAnalysisResult(StrictOutputModel):
    """
    /langchain/analyze-query 的核心返回数据。
    """

    query: str
    analysis: QueryAnalysis
    provider: str
    model_name: str
    mode: AnalysisMode
    fallback_used: bool = False
    error_message: str | None = None
    raw_content: str | None = None
    duration_ms: float = Field(ge=0)


class LangChainChatResult(StrictOutputModel):
    """
    /langchain/chat 的核心返回数据。
    """

    query: str
    answer: str
    provider: str
    model_name: str
    message_count: int = Field(ge=1)
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    usage_metadata: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = Field(ge=0)

