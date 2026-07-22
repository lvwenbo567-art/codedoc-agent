from typing import Dict, List


def extract_query_terms(query: str) -> List[str]:
    """
    从用户 query 中提取关键词。

    当前版本先使用空格切分。
    后续可以扩展为中文分词、BM25、Embedding 检索等。
    """
    query = query.strip().lower()

    if not query:
        raise ValueError("query 不能为空")

    terms = [term for term in query.split() if term]

    if not terms:
        return [query]

    return terms


def score_chunk(query: str, chunk: Dict) -> int:
    """
    根据 query 和 chunk 内容计算一个简单相关性分数。

    当前是关键词检索版本：
    - query 完整出现在 content 中，加 5 分
    - query 完整出现在 source_name 中，加 3 分
    - 每个关键词出现在 content 中，按出现次数加分
    - 每个关键词出现在 source_name 中，加 2 分
    """
    query = query.strip().lower()

    if not query:
        raise ValueError("query 不能为空")

    terms = extract_query_terms(query)

    content = chunk["content"].lower()
    source_name = chunk["source_name"].lower()

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

    scored_chunks = []

    for chunk in chunks:
        score = score_chunk(query, chunk)

        if score > 0:
            item = dict(chunk)
            item["score"] = score
            scored_chunks.append(item)

    scored_chunks.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored_chunks[:top_k]