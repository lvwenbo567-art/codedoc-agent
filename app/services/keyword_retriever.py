from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def tokenize_text(text: str) -> list[str]:
    """
    将英文、代码标识符和中文字符切成 BM25 可用的 token。
    """
    normalized = text.strip().lower()

    if not normalized:
        return []

    return TOKEN_PATTERN.findall(normalized)


def extract_query_terms(query: str) -> List[str]:
    """
    从用户 query 中提取关键词。

    英文、代码标识符按 token 提取；中文在没有分词器时先按单字召回。
    """
    terms = tokenize_text(query)

    if not terms:
        raise ValueError("query 不能为空")

    return terms


def score_chunk(query: str, chunk: Dict) -> int:
    """
    兼容旧版关键词打分逻辑，用于单 chunk 简单打分和旧测试。
    """
    query = query.strip().lower()

    if not query:
        raise ValueError("query 不能为空")

    terms = extract_query_terms(query)

    content = str(chunk["content"]).lower()
    source_name = str(chunk["source_name"]).lower()

    score = 0

    if query in content:
        score += 5

    if query in source_name:
        score += 3

    for term in terms:
        score += content.count(term)

        if term in source_name:
            score += 2

    return score


def _document_tokens(chunk: dict[str, Any]) -> list[str]:
    """
    BM25 的文档内容由正文和少量文件名 token 组成。
    """
    content = str(chunk.get("content") or "")
    source_name = str(chunk.get("source_name") or "")

    return tokenize_text(f"{source_name} {source_name} {content}")


def score_chunks_with_bm25(
    query: str,
    chunks: list[dict[str, Any]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict[str, Any]]:
    """
    使用 BM25 对 chunks 进行关键词相关性打分。

    BM25 会考虑词频、逆文档频率和文档长度，比简单 count 更适合关键词召回。
    """
    query_terms = extract_query_terms(query)

    if not chunks:
        return []

    tokenized_documents = [
        _document_tokens(chunk)
        for chunk in chunks
    ]
    document_lengths = [
        len(tokens)
        for tokens in tokenized_documents
    ]
    average_document_length = (
        sum(document_lengths) / len(document_lengths)
        if document_lengths
        else 0.0
    )

    document_frequency: Counter[str] = Counter()
    for tokens in tokenized_documents:
        document_frequency.update(set(tokens))

    document_count = len(chunks)
    scored_chunks: list[dict[str, Any]] = []

    for chunk, tokens, document_length in zip(
        chunks,
        tokenized_documents,
        document_lengths,
    ):
        if not tokens:
            continue

        term_frequency = Counter(tokens)
        score = 0.0

        for term in query_terms:
            frequency = term_frequency.get(term, 0)

            if frequency <= 0:
                continue

            idf = math.log(
                1
                + (
                    document_count
                    - document_frequency[term]
                    + 0.5
                )
                / (document_frequency[term] + 0.5)
            )
            denominator = frequency + k1 * (
                1
                - b
                + b
                * document_length
                / max(average_document_length, 1.0)
            )
            score += idf * (
                frequency
                * (k1 + 1)
                / denominator
            )

        if score <= 0:
            continue

        item = dict(chunk)
        item["score"] = round(score, 6)
        item["keyword_score_type"] = "bm25"
        scored_chunks.append(item)

    scored_chunks.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored_chunks


def search_chunks(
    query: str,
    chunks: List[Dict],
    top_k: int = 5,
) -> List[Dict]:
    """
    从 chunks 中检索与 query 最相关的 Top-K chunks。
    """
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    return score_chunks_with_bm25(
        query=query,
        chunks=chunks,
    )[:top_k]
