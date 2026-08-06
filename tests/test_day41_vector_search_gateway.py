from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from repositories.vector_store import save_vector_index
from services.vector_search_gateway import search_vector_store


def test_search_vector_store_json_backend_keeps_legacy_shape(tmp_path):
    index_path = tmp_path / "vector_index.json"
    records = [
        {
            "chunk_id": "main.py::chunk_0",
            "source_path": "main.py",
            "source_name": "main.py",
            "source_suffix": ".py",
            "chunk_type": "code",
            "chunk_index": 0,
            "content": "hello world",
            "content_preview": "hello world",
            "length": 11,
            "embedding": [1.0, 0.0, 0.0, 0.0],
        }
    ]
    save_vector_index(
        records=records,
        output_path=str(index_path),
        metadata={
            "index_format_version": "1.0",
            "embedding_provider": "mock",
            "embedding_model": "mock-hash-embedding",
            "dimension": 4,
            "normalized": True,
        },
    )

    result = search_vector_store(
        query="hello",
        project_id=1,
        index_path=str(index_path),
        top_k=1,
        embedding_provider="mock",
        embedding_model="mock-hash-embedding",
        mock_dimension=4,
        backend="json",
        include_content=True,
    )

    assert result["backend"] == "json"
    assert result["project_id"] == 1
    assert result["result_count"] == 1
    assert result["results"][0]["rank"] == 1
    assert result["results"][0]["chunk_id"] == "main.py::chunk_0"
    assert "vector_score" in result["results"][0]
    assert result["results"][0]["content"] == "hello world"
