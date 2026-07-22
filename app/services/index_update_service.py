from utils.batch_utils import iter_batches
from utils.content_hash import compute_content_hash
from clients.embedding_client import EmbeddingClient


def build_vector_record(
    chunk: dict,
    embedding: list[float],
    content_hash: str,
) -> dict:
    """
    将 chunk、embedding 和 content_hash
    组合成向量记录。
    """
    return {
        "chunk_id": chunk["chunk_id"],
        "source_path": chunk["source_path"],
        "source_name": chunk["source_name"],
        "source_suffix": chunk["source_suffix"],
        "chunk_type": chunk["chunk_type"],
        "chunk_index": chunk["chunk_index"],
        "content": chunk["content"],
        "length": chunk["length"],
        "content_hash": content_hash,
        "embedding": embedding,
    }


def validate_reusable_index(
    metadata: dict,
    embedding_provider: str,
    embedding_model: str,
    normalized: bool,
) -> None:
    """
    判断旧索引是否由相同的 Embedding 配置生成。
    """
    if metadata.get(
        "index_format_version"
    ) == "legacy":
        raise ValueError(
            "旧索引没有完整元数据，"
            "不能进行增量更新，请执行全量重建"
        )

    if (
        metadata.get("embedding_provider")
        != embedding_provider
    ):
        raise ValueError(
            "Embedding Provider 已变化，"
            "必须执行全量重建"
        )

    if (
        metadata.get("embedding_model")
        != embedding_model
    ):
        raise ValueError(
            "Embedding 模型已变化，"
            "必须执行全量重建"
        )

    if (
        metadata.get("normalized")
        != normalized
    ):
        raise ValueError(
            "向量归一化配置已变化，"
            "必须执行全量重建"
        )


def build_incremental_records(
    chunks: list[dict],
    old_records: list[dict],
    embedding_client: EmbeddingClient,
    batch_size: int,
) -> tuple[list[dict], dict]:
    """
    增量构建向量记录。

    未变化的 chunk 复用旧 embedding；
    新增或修改的 chunk 重新生成 embedding；
    当前已删除的 chunk 不写入新索引；
    内容完全相同的多个 chunk 只调用一次 Embedding。
    """
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    old_by_chunk_id = {
        record["chunk_id"]: record
        for record in old_records
    }
    current_chunk_ids = {
        chunk["chunk_id"]
        for chunk in chunks
    }
    old_chunk_ids = set(old_by_chunk_id.keys())
    deleted_chunk_ids = old_chunk_ids - current_chunk_ids

    embedding_by_chunk_id: dict[str, list[float]] = {}
    hash_by_chunk_id: dict[str, str] = {}
    pending_by_hash: dict[str, list[dict]] = {}

    reused_count = 0
    new_count = 0
    updated_count = 0

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        content_hash = compute_content_hash(chunk["content"])
        hash_by_chunk_id[chunk_id] = content_hash

        old_record = old_by_chunk_id.get(chunk_id)
        old_hash = None

        if old_record is not None:
            old_hash = old_record.get("content_hash")

        if (
            old_record is not None
            and old_hash == content_hash
            and isinstance(old_record.get("embedding"), list)
        ):
            embedding_by_chunk_id[chunk_id] = old_record["embedding"]
            reused_count += 1
            continue

        if old_record is None:
            new_count += 1
        else:
            updated_count += 1

        pending_by_hash.setdefault(content_hash, []).append(chunk)

    unique_pending_chunks = [
        same_content_chunks[0]
        for same_content_chunks in pending_by_hash.values()
    ]
    hash_to_embedding: dict[str, list[float]] = {}

    for batch in iter_batches(
        items=unique_pending_chunks,
        batch_size=batch_size,
    ):
        contents = [
            chunk["content"]
            for chunk in batch
        ]
        embeddings = embedding_client.embed_texts(contents)

        for chunk, embedding in zip(batch, embeddings):
            content_hash = hash_by_chunk_id[chunk["chunk_id"]]
            hash_to_embedding[content_hash] = embedding

    for content_hash, same_content_chunks in pending_by_hash.items():
        embedding = hash_to_embedding[content_hash]

        for chunk in same_content_chunks:
            embedding_by_chunk_id[chunk["chunk_id"]] = embedding

    new_records = []

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        new_records.append(
            build_vector_record(
                chunk=chunk,
                embedding=embedding_by_chunk_id[chunk_id],
                content_hash=hash_by_chunk_id[chunk_id],
            )
        )

    pending_chunk_count = sum(
        len(group)
        for group in pending_by_hash.values()
    )
    unique_embedding_count = len(unique_pending_chunks)
    duplicate_content_count = pending_chunk_count - unique_embedding_count

    stats = {
        "total_chunk_count": len(chunks),
        "old_record_count": len(old_records),
        "reused_count": reused_count,
        "new_count": new_count,
        "updated_count": updated_count,
        "deleted_count": len(deleted_chunk_ids),
        "embedded_chunk_count": pending_chunk_count,
        "unique_embedding_count": unique_embedding_count,
        "duplicate_content_count": duplicate_content_count,
        "deleted_chunk_ids": sorted(deleted_chunk_ids),
    }

    return new_records, stats
