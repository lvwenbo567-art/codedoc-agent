import json
from pathlib import Path
import sys

import httpx
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from embedding_client import EmbeddingClient, EmbeddingConfig


def test_ollama_embedding_client_posts_to_api_embed():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "model": "nomic-embed-text",
                "embeddings": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    embedding_client = EmbeddingClient(
        config=EmbeddingConfig(
            provider="ollama",
            model_name="nomic-embed-text",
            base_url="http://localhost:11434",
            normalize=False,
        ),
        http_client=client,
    )

    vectors = embedding_client.embed_texts(["hello", "world"])

    assert seen["url"] == "http://localhost:11434/api/embed"
    assert seen["payload"] == {
        "model": "nomic-embed-text",
        "input": ["hello", "world"],
    }
    assert vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_openai_compatible_embedding_client_posts_to_embeddings():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("Authorization")
        seen["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 2.0]},
                    {"index": 0, "embedding": [2.0, 0.0]},
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    embedding_client = EmbeddingClient(
        config=EmbeddingConfig(
            provider="openai_compatible",
            model_name="text-embedding-test",
            base_url="http://localhost:8001/v1",
            api_key="EMPTY",
            normalize=False,
        ),
        http_client=client,
    )

    vectors = embedding_client.embed_texts(["a", "b"])

    assert seen["url"] == "http://localhost:8001/v1/embeddings"
    assert seen["authorization"] == "Bearer EMPTY"
    assert seen["payload"] == {
        "model": "text-embedding-test",
        "input": ["a", "b"],
    }
    assert vectors == [[2.0, 0.0], [0.0, 2.0]]


def test_embedding_client_maps_http_error_to_runtime_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    embedding_client = EmbeddingClient(
        config=EmbeddingConfig(
            provider="openai_compatible",
            model_name="text-embedding-test",
            base_url="http://localhost:8001/v1",
        ),
        http_client=client,
    )

    with pytest.raises(RuntimeError, match="Embedding 模型服务返回状态码"):
        embedding_client.embed_text("hello")


def test_embedding_client_rejects_inconsistent_vector_dimensions():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 1, "embedding": [1.0]},
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    embedding_client = EmbeddingClient(
        config=EmbeddingConfig(
            provider="openai_compatible",
            model_name="text-embedding-test",
            base_url="http://localhost:8001/v1",
        ),
        http_client=client,
    )

    with pytest.raises(ValueError, match="向量维度不一致"):
        embedding_client.embed_texts(["a", "b"])
