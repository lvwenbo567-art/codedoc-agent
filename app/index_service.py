import time
from typing import Dict, List, Optional, Tuple

from batch_utils import split_batches
from chunk_storage import load_chunks_from_json
from config import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    DEFAULT_NORMALIZE_EMBEDDING,
    DEFAULT_VECTOR_INDEX_PATH,
)
from embedding_client import EmbeddingClient, EmbeddingConfig
from vector_store import build_index_metadata, save_vector_index


def build_vector_records(
    chunks: List[Dict],
    embedding_client: EmbeddingClient,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> Tuple[List[Dict], Dict]:
    """
    按 batch_size 分批为 chunks 生成向量记录，并返回建库统计信息。
    """
    if not chunks:
        raise ValueError("chunks 不能为空")

    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    chunk_batches = split_batches(
        items=chunks,
        batch_size=batch_size,
    )

    records = []
    start_time = time.perf_counter()

    for chunk_batch in chunk_batches:
        contents = [
            chunk["content"]
            for chunk in chunk_batch
        ]

        embeddings = embedding_client.embed_texts(contents)

        for chunk, embedding in zip(chunk_batch, embeddings):
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

    duration_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2,
    )
    call_stats = embedding_client.get_call_stats()

    stats = {
        "chunk_count": len(chunks),
        "vector_count": len(records),
        "batch_size": batch_size,
        "batch_count": len(chunk_batches),
        "request_count": call_stats["request_count"],
        "retry_count": call_stats["retry_count"],
        "duration_ms": duration_ms,
    }

    return records, stats


def build_vector_index_from_json(
    chunks_path: str,
    output_path: str = DEFAULT_VECTOR_INDEX_PATH,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: Optional[str] = None,
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL,
    embedding_api_key: str = DEFAULT_EMBEDDING_API_KEY,
    timeout_seconds: float = DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    mock_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    normalize: bool = DEFAULT_NORMALIZE_EMBEDDING,
    # Day19/20 compatibility aliases.
    model_name: Optional[str] = None,
    dimension: Optional[int] = None,
) -> Dict:
    """
    从 chunks.json 构建带元数据和建库统计的向量索引。
    """
    if model_name is not None and embedding_model is None:
        embedding_model = model_name

    if dimension is not None:
        mock_dimension = dimension

    embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL

    chunks = load_chunks_from_json(chunks_path)

    config = EmbeddingConfig(
        provider=embedding_provider,
        model_name=embedding_model,
        base_url=embedding_base_url,
        api_key=embedding_api_key,
        timeout_seconds=timeout_seconds,
        mock_dimension=mock_dimension,
        normalize=normalize,
    )

    client = EmbeddingClient(config=config)

    records, build_stats = build_vector_records(
        chunks=chunks,
        embedding_client=client,
        batch_size=batch_size,
    )

    actual_dimension = len(records[0]["embedding"])

    metadata = build_index_metadata(
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        dimension=actual_dimension,
        normalized=normalize,
        record_count=len(records),
        build_stats=build_stats,
    )

    saved_path = save_vector_index(
        records=records,
        output_path=output_path,
        metadata=metadata,
    )

    return {
        "chunks_path": chunks_path,
        "output_path": str(saved_path),
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "model_name": embedding_model,
        "dimension": actual_dimension,
        "chunk_count": len(chunks),
        "vector_count": len(records),
        "build_stats": build_stats,
        "index_metadata": metadata,
    }
