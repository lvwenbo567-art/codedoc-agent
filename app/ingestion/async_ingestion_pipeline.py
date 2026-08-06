from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clients.async_embedding_client import AsyncEmbeddingClient
from clients.embedding_client import EmbeddingClient, EmbeddingConfig
from ingestion.chunker import build_chunks
from ingestion.file_loader import read_text_file, scan_project_files
from ingestion.ingestion_job_models import IngestionStage
from ingestion.ingestion_job_runner import ProgressCallback
from utils.document_schema import ProjectFile
from vectorstores.async_base import AsyncVectorStore
from vectorstores.models import VectorPoint


@dataclass(frozen=True)
class AsyncIngestionDependencies:
    scan_project: Callable[[str], list[Any]]
    load_files: Callable[[list[Any]], list[dict[str, Any]]]
    parse_files: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
    build_chunks: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
    embed_texts: Callable[[list[str], int], Any]
    vector_store: AsyncVectorStore


class AsyncCodeDocIngestionPipeline:
    """将既有同步扫描/切分算法与异步模型、异步向量库组合。"""

    def __init__(self, *, dependencies: AsyncIngestionDependencies) -> None:
        self.dependencies = dependencies

    async def run(self, *, job_id: str, request_data: dict[str, Any], progress_callback: ProgressCallback) -> dict[str, Any]:
        project_id = int(request_data["project_id"])
        root = str(request_data["project_root"])
        await progress_callback(IngestionStage.SCANNING, 0.05, None)
        scanned = await asyncio.to_thread(self.dependencies.scan_project, root)
        await progress_callback(IngestionStage.LOADING, 0.15, {"scanned_file_count": len(scanned)})
        loaded = await asyncio.to_thread(self.dependencies.load_files, scanned)
        await progress_callback(IngestionStage.PARSING, 0.30, {"loaded_file_count": len(loaded)})
        parsed = await asyncio.to_thread(self.dependencies.parse_files, loaded)
        await progress_callback(IngestionStage.CHUNKING, 0.45, None)
        chunks = await asyncio.to_thread(self.dependencies.build_chunks, parsed)
        if not chunks:
            raise ValueError("没有生成任何 Chunk")
        await progress_callback(IngestionStage.EMBEDDING, 0.55, {"chunk_count": len(chunks)})
        texts = [str(chunk.get("content") or "") for chunk in chunks]
        vectors = await self.dependencies.embed_texts(
            texts,
            int(request_data.get("embedding_batch_size", 32)),
        )
        if len(vectors) != len(chunks):
            raise ValueError("Embedding 数量和 Chunk 数量不一致")
        points = [VectorPoint.from_index_record(project_id=project_id, record={**chunk, "embedding": vector})
                  for chunk, vector in zip(chunks, vectors)]
        await progress_callback(IngestionStage.UPSERTING, 0.85, {"vector_count": len(points)})
        upsert = await self.dependencies.vector_store.upsert(
            points=points, batch_size=int(request_data.get("upsert_batch_size", 128))
        )
        await progress_callback(IngestionStage.COMPLETED, 1.0, None)
        return {"job_id": job_id, "project_id": project_id, "scanned_file_count": len(scanned),
                "loaded_file_count": len(loaded), "chunk_count": len(chunks), "vector_count": len(points),
                "upserted_count": upsert.upserted_count}


def _load_paths(paths: list[Path]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in paths:
        content = read_text_file(path)
        files.append(ProjectFile(path=str(path), name=path.name, suffix=path.suffix.lower(),
                                 content=content, length=len(content)).to_dict())
    return files


def build_default_async_ingestion_pipeline(*, vector_store: AsyncVectorStore,
                                           async_embedding_client: AsyncEmbeddingClient | None,
                                           embedding_config: EmbeddingConfig) -> AsyncCodeDocIngestionPipeline:
    """用项目既有扫描与切块逻辑构建 Day42 Job Runner。"""
    async def embed_texts(
        texts: list[str],
        batch_size: int,
    ) -> list[list[float]]:
        if embedding_config.provider == "mock":
            client = EmbeddingClient(config=embedding_config)
            vectors: list[list[float]] = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                vectors.extend(
                    await asyncio.to_thread(
                        client.embed_texts,
                        batch,
                    )
                )
            return vectors
        if async_embedding_client is None:
            raise RuntimeError("真实 Embedding Client 尚未初始化")
        return await async_embedding_client.embed_texts(
            texts,
            batch_size=batch_size,
        )

    return AsyncCodeDocIngestionPipeline(
        dependencies=AsyncIngestionDependencies(
            scan_project=scan_project_files,
            load_files=_load_paths,
            # Python AST 解析实际发生于 build_chunks 的 build_python_code_chunks；该阶段保留显式进度节点。
            parse_files=lambda files: files,
            build_chunks=build_chunks,
            embed_texts=embed_texts,
            vector_store=vector_store,
        )
    )
