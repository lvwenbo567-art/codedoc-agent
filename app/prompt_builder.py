from typing import Dict, List

from config import DEFAULT_MAX_CONTEXT_CHARS


def build_context(
    retrieved_chunks: List[Dict],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    """
    将检索结果构建为带来源编号的上下文。
    """
    if max_context_chars <= 0:
        raise ValueError("max_context_chars 必须大于 0")

    context_blocks = []
    current_length = 0

    for source_number, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        content = chunk.get("content", "").strip()

        if not content:
            continue

        block = (
            f"[Source {source_number}]\n"
            f"文件：{chunk['source_path']}\n"
            f"类型：{chunk['chunk_type']}\n"
            f"Chunk ID：{chunk['chunk_id']}\n"
            f"内容：\n{content}"
        )

        remaining = max_context_chars - current_length

        if remaining <= 0:
            break

        if len(block) > remaining:
            block = block[:remaining]

        context_blocks.append(block)
        current_length += len(block)

    return "\n\n".join(context_blocks)


def build_rag_prompt(
    query: str,
    retrieved_chunks: List[Dict],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    """
    构建基础 RAG Prompt。
    """
    if not query or not query.strip():
        raise ValueError("query 不能为空")

    context = build_context(
        retrieved_chunks=retrieved_chunks,
        max_context_chars=max_context_chars,
    )

    if not context:
        context = "没有检索到可以支持回答的项目内容。"

    return f"""
你是 CodeDoc Research Agent，一个用于理解代码项目的助手。

请严格根据下面的检索上下文回答用户问题。

回答要求：
1. 不要编造上下文中没有的信息。
2. 如果上下文不足，请明确说明信息不足。
3. 使用清晰、准确的语言解释。
4. 引用信息时使用 [Source 1]、[Source 2] 等来源编号。
5. 如果问题涉及代码，请说明相关文件和代码作用。

【用户问题】
{query}

【检索上下文】
{context}

【回答】
""".strip()