import json
from pathlib import Path
import sys

import httpx
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from embedding_client import EmbeddingClient, EmbeddingConfig


def test_embedding_retry_success_after_500():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return httpx.Response(500, json={"error": "temporary error"})

        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "index": 0,
                        "embedding": [1.0, 0.0],
                    }
                ]
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = EmbeddingClient(
        config=EmbeddingConfig(
            provider="openai_compatible",
            model_name="test-embedding",
            base_url="http://localhost:8001/v1",
            retry_backoff_seconds=0,
            max_retries=2,
            normalize=False,
        ),
        http_client=http_client,
    )

    vector = client.embed_text("hello")

    assert vector == [1.0, 0.0]
    assert client.request_count == 2
    assert client.retry_count == 1


def test_embedding_400_does_not_retry():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad request"})

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = EmbeddingClient(
        config=EmbeddingConfig(
            provider="openai_compatible",
            model_name="test-embedding",
            base_url="http://localhost:8001/v1",
            retry_backoff_seconds=0,
            max_retries=2,
        ),
        http_client=http_client,
    )

    with pytest.raises(RuntimeError, match="状态码：400"):
        client.embed_text("hello")

    assert client.request_count == 1
    assert client.retry_count == 0


def test_embedding_retry_sends_same_payload():
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content.decode("utf-8")))

        if len(payloads) == 1:
            return httpx.Response(429, json={"error": "rate limited"})

        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "index": 0,
                        "embedding": [0.0, 1.0],
                    }
                ]
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = EmbeddingClient(
        config=EmbeddingConfig(
            provider="openai_compatible",
            model_name="test-embedding",
            base_url="http://localhost:8001/v1",
            retry_backoff_seconds=0,
            max_retries=1,
            normalize=False,
        ),
        http_client=http_client,
    )

    vector = client.embed_text("hello")

    assert vector == [0.0, 1.0]
    assert payloads == [
        {"model": "test-embedding", "input": ["hello"]},
        {"model": "test-embedding", "input": ["hello"]},
    ]
    assert client.get_call_stats() == {
        "request_count": 2,
        "retry_count": 1,
    }
