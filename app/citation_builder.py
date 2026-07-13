from typing import Dict, List


def build_citations(
    retrieved_chunks: List[Dict],
) -> List[Dict]:
    """
    根据检索结果构建结构化引用信息。
    """
    citations = []

    for source_number, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        citations.append(
            {
                "citation_id": f"Source {source_number}",
                "rank": chunk["rank"],
                "chunk_id": chunk["chunk_id"],
                "source_path": chunk["source_path"],
                "source_name": chunk["source_name"],
                "source_suffix": chunk["source_suffix"],
                "chunk_type": chunk["chunk_type"],
                "chunk_index": chunk["chunk_index"],
                "score": chunk["score"],
            }
        )

    return citations