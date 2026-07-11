from typing import Dict, List

from chunk_storage import load_chunks_from_json
from config import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_INDEX_PATH,
)
from embedding_client import EmbeddingClient
from vector_store import save_vector_index


def build_vector_records(
    chunks: List[Dict],
    embedding_client: EmbeddingClient,
) -> List[Dict]:
    """
    为 chunks 生成向量记录。
    """
    records = []

    for chunk in chunks:
        embedding = embedding_client.embed_text(chunk["content"])

        records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source_path": chunk["source_path"],
                "source_name": chunk["source_name"],
                "source_suffix": chunk["source_suffix"],
                "chunk_type": chunk["chunk_type"],
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "length": chunk["length"],
                "embedding": embedding,
            }
        )

    return records


def build_vector_index_from_json(
    chunks_path: str,
    output_path: str = DEFAULT_VECTOR_INDEX_PATH,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
) -> Dict:
    """
    从 chunks.json 构建向量索引。
    """
    chunks = load_chunks_from_json(chunks_path)

    embedding_client = EmbeddingClient(
        model_name=model_name,
        dimension=dimension,
    )

    records = build_vector_records(
        chunks=chunks,
        embedding_client=embedding_client,
    )

    saved_path = save_vector_index(
        records=records,
        output_path=output_path,
    )

    return {
        "chunks_path": chunks_path,
        "output_path": str(saved_path),
        "model_name": model_name,
        "dimension": dimension,
        "chunk_count": len(chunks),
        "vector_count": len(records),
    }