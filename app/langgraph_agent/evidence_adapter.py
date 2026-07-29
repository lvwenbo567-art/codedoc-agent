from __future__ import annotations

import json
from typing import Any

from langgraph_agent.state import GraphEvidence
'''

工具返回可能是：
{
    "results": [
        {
            "chunk_id": "c1",
            "source_path": "test_project/search.py",
            "content": "def keyword_score...",
            "score": 0.9,
        }
    ]
}
转成 Graph 统一证据：
{
    "chunk_id": "c1",
    "source_path": "test_project/search.py",
    "evidence_type": "code_search",
    "content": "def keyword_score...",
    "score": 0.9,
    "metadata": 原始结果,
}
这一层是给 LangGraph State 用的：
state["evidence"]

Graph 里的 evidence 再转成回答层能用的 chunk：
{
    "rank": 1,
    "chunk_id": "c1",
    "source_path": "test_project/search.py",
    "source_name": "search.py",
    "chunk_type": "code",
    "score": 0.9,
    "content": "def keyword_score...",
}
这一层是给已有 RAG 回答模块用的：
prompt_builder
citation_builder
answer_quality
因为你之前的回答链路本来就是基于 chunks 设计的。
'''

RESULT_LIST_KEYS = (
    "results",
    "retrieved_chunks",
    "items",
    "matches",
    "chunks",
    "files",
    "entries",
)


def _safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    except TypeError:
        return str(value)


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        dumped = value.model_dump()

        if isinstance(dumped, dict):
            return dumped

    return {"content": str(value)}


def extract_result_items(data: Any) -> list[dict[str, Any]]:
    '''这个函数负责从工具返回的 data 中提取结果列表。'''
    if data is None:
        return []

    if isinstance(data, list):
        return [
            _as_mapping(item)
            for item in data
        ]

    mapping = _as_mapping(data)

    for key in RESULT_LIST_KEYS:
        items = mapping.get(key)

        if isinstance(items, list):
            return [
                _as_mapping(item)
                for item in items
            ]

    return [mapping]


def _extract_content(item: dict[str, Any]) -> str:
    '''这个函数从 item 中提取证据正文。'''
    content_like_keys = {
        "content",
        "text",
        "snippet",
        "code",
        "document",
    }
    for key in (
        "content",
        "text",
        "snippet",
        "code",
        "document",
    ):
        value = item.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    if set(item.keys()) <= content_like_keys:
        return ""

    return _safe_json_dumps(item)


def _has_empty_explicit_content(
    item: dict[str, Any],
) -> bool:
    content_keys = (
        "content",
        "text",
        "snippet",
        "code",
        "document",
    )

    existing_values = [
        item.get(key)
        for key in content_keys
        if key in item
    ]

    if not existing_values:
        return False

    return not any(
        isinstance(value, str)
        and value.strip()
        for value in existing_values
    )


def _extract_score(item: dict[str, Any]) -> float | None:
    for key in (
        "rerank_score",
        "final_score",
        "score",
        "vector_score",
        "keyword_score",
        "match_score",
    ):
        value = item.get(key)

        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def convert_item_to_evidence(
    *,
    item: dict[str, Any],
    evidence_type: str,
) -> GraphEvidence | None:#它负责把一条工具结果 item 转成一条 GraphEvidence
    if (
        evidence_type != "project_structure"
        and _has_empty_explicit_content(item)
    ):
        return None

    content = _extract_content(item)

    if not content.strip():
        return None

    source_path = str(
        item.get("source_path")
        or item.get("path")
        or item.get("file_path")
        or item.get("source_name")
        or (
            "<project-structure>"
            if evidence_type == "project_structure"
            else "<unknown-source>"
        )
    )
    chunk_id_value = item.get("chunk_id")

    return GraphEvidence(
        chunk_id=str(chunk_id_value) if chunk_id_value is not None else None,
        source_path=source_path,
        source_name=(
            str(item["source_name"])
            if item.get("source_name") is not None
            else None
        ),
        chunk_type=(
            str(item["chunk_type"])
            if item.get("chunk_type") is not None
            else None
        ),
        evidence_type=evidence_type,
        content=content,
        score=_extract_score(item),
        symbol_name=(
            str(item["symbol_name"])
            if item.get("symbol_name") is not None
            else None
        ),
        qualified_name=(
            str(item["qualified_name"])
            if item.get("qualified_name") is not None
            else None
        ),
        start_line=_optional_int(item.get("start_line")),
        end_line=_optional_int(item.get("end_line")),
        metadata=item,
    )


def convert_result_to_evidence(
    *,
    data: Any,
    evidence_type: str,
) -> list[GraphEvidence]:#这个函数把整个工具 data 转成 evidence 列表。
    evidence: list[GraphEvidence] = []

    for item in extract_result_items(data):
        converted = convert_item_to_evidence(
            item=item,
            evidence_type=evidence_type,
        )

        if converted is not None:
            evidence.append(converted)

    return evidence


def evidence_to_answer_chunks(
    evidence: list[GraphEvidence],
) -> list[dict[str, Any]]:#这个函数把 GraphEvidence 转成 AnswerService 能用的 chunks 格式。
    chunks: list[dict[str, Any]] = []

    for rank, item in enumerate(evidence, start=1):
        source_path = item.get("source_path") or "<unknown-source>"
        chunk_id = item.get("chunk_id") or f"{source_path}::evidence::{rank}"

        chunks.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "source_path": source_path,
                "source_name": (
                    item.get("source_name")
                    or str(source_path).replace("\\", "/").split("/")[-1]
                ),
                "source_suffix": "",
                "chunk_type": item.get("chunk_type") or item.get("evidence_type") or "unknown",
                "chunk_index": rank - 1,
                "score": item.get("score") if item.get("score") is not None else 0.0,
                "content": item.get("content") or "",
            }
        )

    return chunks


def evidence_to_retrieved_chunks(
    evidence: list[GraphEvidence],
) -> list[dict[str, Any]]:
    return evidence_to_answer_chunks(
        evidence
    )


def compress_project_structure_evidence(
    evidence: list[GraphEvidence],
    max_entries: int = 80,
) -> list[GraphEvidence]:#这个函数专门处理项目结构证据。
    if not evidence:
        return []

    directories = []
    files = []

    for item in evidence:
        metadata = item.get("metadata") or {}
        entry_type = metadata.get("type")

        if entry_type == "directory":
            directories.append(item)
        elif entry_type == "file":
            files.append(item)

    def format_line(item: GraphEvidence) -> str:
        metadata = item.get("metadata") or {}
        path = metadata.get("path") or item.get("source_path")
        entry_type = metadata.get("type") or "unknown"
        depth = metadata.get("depth")
        return f"- [{entry_type}] {path} (depth={depth})"

    directory_lines = [
        format_line(item)
        for item in directories[:max_entries]
    ]
    remaining = max(0, max_entries - len(directory_lines))
    file_lines = [
        format_line(item)
        for item in files[:remaining]
    ]

    content = "\n".join(
        [
            "项目结构摘要：",
            f"- 总条目数：{len(evidence)}",
            f"- 目录数：{len(directories)}",
            f"- 文件数：{len(files)}",
            "",
            "主要目录：",
            *(directory_lines or ["- 暂无目录条目"]),
            "",
            "关键文件示例：",
            *(file_lines or ["- 暂无文件条目"]),
        ]
    )

    return [
        GraphEvidence(
            chunk_id=None,
            source_path="<project-structure-summary>",
            source_name="project_structure",
            chunk_type="project_structure",
            evidence_type="project_structure",
            content=content,
            score=None,
            symbol_name=None,
            qualified_name=None,
            start_line=None,
            end_line=None,
            metadata={
                "original_entry_count": len(evidence),
                "directory_count": len(directories),
                "file_count": len(files),
                "included_entry_count": len(directory_lines) + len(file_lines),
                "max_summary_entries": max_entries,
            },
        )
    ]
