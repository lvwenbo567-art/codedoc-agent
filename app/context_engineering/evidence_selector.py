from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from context_engineering.token_counter import TokenCounter

'''
1. 分数足够靠前；
2. 总数量未达到 max_items；
3. chunk_id 没出现过；
4. content_hash 没出现过；
5. 所在文件入选数量未超过 max_items_per_source；
6. 截断后单条长度不超过 max_single_tokens；
7. 加入后总证据 Token 不超过 max_total_tokens。
'''
@dataclass(frozen=True)
class EvidenceSelectionResult:
    selected: list[dict[str, Any]]
    selected_tokens: int
    dropped_count: int
    truncated_count: int


def extract_evidence_score(evidence: dict[str, Any]) -> float:
    for field in (
        "rerank_score", "final_score", "multi_query_score", "score",
        "vector_score", "keyword_score",
    ):
        try:
            if evidence.get(field) is not None:
                return float(evidence[field])
        except (TypeError, ValueError):
            continue
    return 0.0


def truncate_text_by_tokens(
    *, text: str, max_tokens: int, token_counter: TokenCounter
) -> tuple[str, bool]:
    """二分寻找不超过 Token 上限的最长字符前缀。"""
    if max_tokens <= 0:
        raise ValueError("max_tokens 必须大于 0")
    if token_counter.count_text(text) <= max_tokens:
        return text, False
    marker = "\n...[truncated]"
    marker_tokens = token_counter.count_text(marker)
    content_budget = max_tokens - marker_tokens
    if content_budget <= 0:
        # 预算过小时优先满足硬上限，不再附加截断标记。
        marker = ""
        content_budget = max_tokens

    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if token_counter.count_text(text[:middle]) <= content_budget:
            low = middle
        else:
            high = middle - 1
    return text[:low] + marker, True


def select_evidence(
    *,
    evidence_items: list[dict[str, Any]],
    token_counter: TokenCounter,
    max_total_tokens: int,
    max_single_tokens: int,
    max_items: int,
    max_items_per_source: int,
) -> EvidenceSelectionResult:
    """按分数选择去重、多来源且符合 Token 预算的证据。"""
    if min(max_total_tokens, max_single_tokens, max_items, max_items_per_source) <= 0:
        raise ValueError("Evidence 预算必须大于 0")

    seen_chunk_ids: set[str] = set()#同一个 Chunk 不重复
    seen_content_hashes: set[str] = set()#即使 chunk_id 不同，内容相同也不重复
    source_counts: dict[str, int] = defaultdict(int)#同一个文件不占满所有位置
    selected: list[dict[str, Any]] = []#最终结果
    used_tokens = 0#已使用 Token
    truncated_count = 0#截断统计

    for item in sorted(evidence_items, key=extract_evidence_score, reverse=True):
        if len(selected) >= max_items:
            break
        chunk_id = str(item.get("chunk_id") or "")
        content_hash = str(item.get("content_hash") or "")
        source_path = str(item.get("source_path") or "unknown")
        if chunk_id and chunk_id in seen_chunk_ids:
            continue
        if content_hash and content_hash in seen_content_hashes:
            continue
        if source_counts[source_path] >= max_items_per_source:
            continue

        content, truncated = truncate_text_by_tokens(
            text=str(item.get("content") or ""),
            max_tokens=max_single_tokens,
            token_counter=token_counter,
        )
        item_tokens = token_counter.count_text(content)
        if used_tokens + item_tokens > max_total_tokens:
            continue

        selected_item = dict(item)
        selected_item["content"] = content
        selected_item["context_truncated"] = truncated
        selected.append(selected_item)
        used_tokens += item_tokens
        source_counts[source_path] += 1
        if chunk_id:
            seen_chunk_ids.add(chunk_id)
        if content_hash:
            seen_content_hashes.add(content_hash)
        truncated_count += int(truncated)

    return EvidenceSelectionResult(
        selected=selected,
        selected_tokens=used_tokens,
        dropped_count=len(evidence_items) - len(selected),
        truncated_count=truncated_count,
    )
