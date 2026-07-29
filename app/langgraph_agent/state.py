from __future__ import annotations

from typing import Annotated, Any, Literal
'''
Annotated：
    给 State 字段绑定 Reducer。

'''
from typing_extensions import TypedDict

from langgraph_agent.reducers import (
    merge_evidence,
    merge_unique_strings,
)


QueryType = Literal[
    "code",
    "document",
    "structure",
    "unknown",
]


class GraphEvidence(
    TypedDict,
    total=False,#这些字段不是每个 evidence 都必须有。
):
    """
    LangGraph 工作流内部统一证据格式。
    """

    chunk_id: str | None
    source_path: str
    source_name: str | None
    chunk_type: str | None
    evidence_type: str
    content: str
    score: float | None
    symbol_name: str | None
    qualified_name: str | None
    start_line: int | None
    end_line: int | None
    metadata: dict[str, Any]


class GraphCitation(
    TypedDict,
    total=False,
):
    citation_id: str
    source_path: str
    chunk_id: str | None
    score: float | None
    start_line: int | None
    end_line: int | None


class CodeDocGraphState(
    TypedDict,
    total=False,
):
    """
    Day35 CodeDoc Workflow 的共享 State。

    Node 读取整个 State，但只返回自己修改的部分字段；
    LangGraph 会根据 Reducer 合并这些字段。
    """

    query: str
    project_id: int
    query_type: QueryType
    evidence: Annotated[
        list[GraphEvidence],
        merge_evidence,
    ]
    execution_steps: Annotated[list[str], merge_unique_strings]
    degrade_reasons: Annotated[list[str], merge_unique_strings]
    citations: list[GraphCitation]
    answer: str
    evidence_sufficient: bool
    error_message: str | None
    retrieval_strategy: Literal[
        "original",
        "multi_query",
        "structure",
        "none",
    ]
    symbol_name: str | None
    query_decision: dict[str, Any]
    evidence_assessment: dict[str, Any]
    retrieval_metadata: dict[str, Any]
    answer_quality: dict[str, Any]
    degraded: bool
