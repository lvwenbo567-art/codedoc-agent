"""
测试项目检索模块。

这个文件用于测试关键词检索、Top-K 排序和简单过滤逻辑。
"""


def keyword_score(query: str, text: str) -> int:
    """
    根据 query 中的关键词在文本中出现的次数计算简单分数。
    """
    terms = [
        term.lower()
        for term in query.split()
        if term.strip()
    ]

    lower_text = text.lower()

    return sum(
        lower_text.count(term)
        for term in terms
    )


def search_documents(query: str, documents: list[dict], top_k: int = 3) -> list[dict]:
    """
    对文档列表进行关键词检索，并返回 Top-K 结果。
    """
    scored = []

    for document in documents:
        score = keyword_score(
            query=query,
            text=document["content"],
        )

        if score > 0:
            scored.append(
                {
                    "document_id": document["document_id"],
                    "score": score,
                    "content": document["content"],
                }
            )

    scored.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored[:top_k]
