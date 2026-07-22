from typing import Dict, List

from config import DEFAULT_MAX_CONTEXT_CHARS


SYSTEM_PROMPT = """
你是 CodeDoc Research Agent，一个用于分析代码项目与项目文档的助手。

回答要求：
1. 严格根据检索上下文回答。
2. 不要编造上下文中不存在的信息。
3. 证据不足时明确说明信息不足。
4. 使用 [Source 1] 形式标注来源。
5. 不允许伪造不存在的 Source。
6. 涉及代码时说明文件路径和代码作用。
""".strip()


def build_context(
    retrieved_chunks: List[Dict],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    """
    将检索到的 chunks 组装成带 Source 编号的上下文。
    """
    if max_context_chars <= 0:
        raise ValueError("max_context_chars 必须大于 0")

    blocks = []
    current_length = 0

    for source_number, chunk in enumerate(retrieved_chunks, start=1):
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

        blocks.append(block)
        current_length += len(block)

    return "\n\n".join(blocks)


def build_rag_user_prompt(
    query: str,
    retrieved_chunks: List[Dict],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    """
    构建 RAG 问答的 user prompt。
    """
    if not query or not query.strip():
        raise ValueError("query 不能为空")

    context = build_context(
        retrieved_chunks=retrieved_chunks,
        max_context_chars=max_context_chars,
    )

    if not context:
        context = "当前没有检索到可以支持回答的项目内容。"

    return f"""
【用户问题】
{query}

【检索上下文】
{context}

请根据以上项目内容回答。
""".strip()


def build_rag_messages(
    query: str,
    retrieved_chunks: List[Dict],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> List[Dict[str, str]]:
    """
    构建标准 Chat messages，供 ChatClient 调用。
    """
    user_prompt = build_rag_user_prompt(
        query=query,
        retrieved_chunks=retrieved_chunks,
        max_context_chars=max_context_chars,
    )

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def build_rag_prompt(
    query: str,
    retrieved_chunks: List[Dict],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    """
    保留旧版字符串 prompt 接口，兼容早期测试和调用方式。
    """
    messages = build_rag_messages(
        query=query,
        retrieved_chunks=retrieved_chunks,
        max_context_chars=max_context_chars,
    )

    return f"{messages[0]['content']}\n\n{messages[1]['content']}"
