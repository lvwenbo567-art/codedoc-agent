from __future__ import annotations

from typing import Any


EvidenceItem = dict[str, Any]


def _build_evidence_key(
    evidence: EvidenceItem,
) -> tuple:
    """
    构造证据去重键。

    优先使用 chunk_id；项目结构这类没有 chunk_id 的证据，
    使用 source_path、evidence_type 和 content 组合去重。
    """
    chunk_id = evidence.get("chunk_id")

    if chunk_id:
        return (
            "chunk_id",
            str(chunk_id),
        )

    qualified_name = evidence.get(
        "qualified_name"
    )

    if qualified_name:
        return (
            "symbol",
            str(
                evidence.get(
                    "source_path"
                )
                or ""
            ),
            str(qualified_name),
        )

    return (
        "content",
        str(evidence.get("source_path") or ""),
        str(evidence.get("evidence_type") or ""),
        str(evidence.get("content") or ""),
    )


def merge_evidence(
    current: list[EvidenceItem] | None,
    new: list[EvidenceItem] | None,
) -> list[EvidenceItem]:
    """
    合并 Graph evidence 列表并去重。

    LangGraph 节点通常只返回部分 State 更新。evidence 字段设置
    Reducer 后，新旧证据会被合并，而不是简单覆盖。
    """
    current_items = current or []
    new_items = new or []

    merged: list[EvidenceItem] = []
    seen_keys: set[tuple] = set()
    '''
    这里的 * 是列表解包。

假设：

current_items = [evidence_1, evidence_2]
new_items = [evidence_3]

那么：

[
    *current_items,
    *new_items,
]

结果就是：

[
    evidence_1,
    evidence_2,
    evidence_3,
]
    '''
    for evidence in [
        *current_items,
        *new_items,
    ]:
        if not isinstance(evidence, dict):
            continue

        key = _build_evidence_key(evidence)

        if key in seen_keys:
            continue

        seen_keys.add(key)
        merged.append(evidence)

    return merged


def merge_unique_strings(
    current: list[str] | None,
    new: list[str] | None,
) -> list[str]:
    merged: list[str] = []

    for item in [
        *(current or []),
        *(new or []),
    ]:
        value = str(item).strip()

        if not value:
            continue

        if value in merged:
            continue

        merged.append(value)

    return merged
