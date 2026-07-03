from typing import Dict, List

from chunk_storage import load_chunks_from_json
from retriever import search_chunks

def build_search_results(
    query: str,
    retrieved_chunks: List[Dict],
) -> List[Dict]:
     """
    将 retriever 返回的 chunks 转换为统一的检索结果结构。
    """
     results = []

     for rank, chunk in enumerate(retrieved_chunks, start=1):
        result = {
            "query": query,
            "rank": rank,
            "score": chunk["score"],
            "chunk_id": chunk["chunk_id"],
            "source_path": chunk["source_path"],
            "source_name": chunk["source_name"],
            "source_suffix": chunk["source_suffix"],
            "chunk_type": chunk["chunk_type"],
            "chunk_index": chunk["chunk_index"],
            "content": chunk["content"],
            "content_preview": chunk["content"][:150],
            "length": chunk["length"],
        }

        results.append(result)

     return results

def search_chunks_from_json(
    input_path: str,
    query: str,
    top_k: int = 5,
) -> List[Dict]:
    """
    从 chunks JSON 文件中读取 chunks，并执行 Top-K 检索。
    """
    chunks = load_chunks_from_json(input_path)

    retrieved_chunks = search_chunks(
        query=query,
        chunks=chunks,
        top_k=top_k,
    )

    results = build_search_results(
        query=query,
        retrieved_chunks=retrieved_chunks,
    )

    return results