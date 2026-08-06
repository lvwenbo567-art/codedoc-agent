from __future__ import annotations

import pytest

from ingestion.async_ingestion_pipeline import AsyncCodeDocIngestionPipeline, AsyncIngestionDependencies
from ingestion.ingestion_job_models import IngestionStage
from vectorstores.models import VectorUpsertResult


class Store:
    async def upsert(self, *, points, batch_size=128):
        self.points = points
        return VectorUpsertResult(len(points), len(points), 1)


@pytest.mark.asyncio
async def test_pipeline_reports_stages_and_writes_vectors() -> None:
    stages = []
    received_batch_sizes = []
    async def progress(stage, value, metadata): stages.append((stage, value))
    async def embed_texts(texts, batch_size):
        received_batch_sizes.append(batch_size)
        return await _vectors(texts)

    pipeline = AsyncCodeDocIngestionPipeline(dependencies=AsyncIngestionDependencies(
        scan_project=lambda _: ["f"], load_files=lambda _: [{"content": "hello", "path": "a.md", "name": "a.md", "suffix": ".md"}],
        parse_files=lambda files: files, build_chunks=lambda _: [{"chunk_id": "a", "content": "hello", "source_path": "a.md", "source_name": "a.md", "source_suffix": ".md"}],
        embed_texts=embed_texts, vector_store=Store(),
    ))
    result = await pipeline.run(
        job_id="job",
        request_data={
            "project_id": 1,
            "project_root": ".",
            "embedding_batch_size": 7,
        },
        progress_callback=progress,
    )
    assert result["upserted_count"] == 1
    assert received_batch_sizes == [7]
    assert [stage for stage, _ in stages] == list(IngestionStage)[1:]


async def _vectors(texts): return [[1.0, 0.0] for _ in texts]
