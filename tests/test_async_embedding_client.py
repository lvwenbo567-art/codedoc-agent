from __future__ import annotations

import pytest

from clients.async_embedding_client import AsyncEmbeddingClient, AsyncEmbeddingConfig
from runtime.async_call_policy import AsyncCallRetryExhaustedError


class Gateway:
    def __init__(self) -> None: self.calls: list[dict] = []
    async def post_json(self, **kwargs):
        self.calls.append(kwargs)
        items = kwargs["payload"]["input"]
        return {"data": [{"index": index, "embedding": [float(index), 1.0]} for index in reversed(range(len(items)))]}


@pytest.mark.asyncio
async def test_embedding_batches_and_restores_response_order() -> None:
    gateway = Gateway()
    client = AsyncEmbeddingClient(config=AsyncEmbeddingConfig(base_url="http://model/v1", model_name="m", batch_size=2), gateway=gateway)
    vectors = await client.embed_texts(["a", "b", "c"])
    assert vectors == [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]
    assert len(gateway.calls) == 2


@pytest.mark.asyncio
async def test_embedding_call_can_override_configured_batch_size() -> None:
    gateway = Gateway()
    client = AsyncEmbeddingClient(
        config=AsyncEmbeddingConfig(
            base_url="http://model/v1",
            model_name="m",
            batch_size=32,
        ),
        gateway=gateway,
    )
    await client.embed_texts(["a", "b", "c"], batch_size=1)
    assert len(gateway.calls) == 3


@pytest.mark.asyncio
async def test_embedding_rejects_empty_text() -> None:
    client = AsyncEmbeddingClient(config=AsyncEmbeddingConfig(base_url="http://model", model_name="m"), gateway=Gateway())
    with pytest.raises(ValueError, match="空文本"):
        await client.embed_texts(["ok", " "])


@pytest.mark.asyncio
async def test_embedding_rejects_inconsistent_vector_dimensions() -> None:
    class DimensionGateway:
        async def post_json(self, **kwargs):
            return {
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 1, "embedding": [1.0, 0.0, 0.0]},
                ]
            }

    client = AsyncEmbeddingClient(
        config=AsyncEmbeddingConfig(
            base_url="http://model/v1",
            model_name="m",
        ),
        gateway=DimensionGateway(),
    )
    with pytest.raises(ValueError, match="维度不一致"):
        await client.embed_texts(["a", "b"])


def test_embedding_config_rejects_unsupported_provider() -> None:
    with pytest.raises(ValueError, match="只支持"):
        AsyncEmbeddingConfig(
            base_url="http://model",
            model_name="m",
            provider="unknown",
        )


@pytest.mark.asyncio
async def test_ollama_nan_response_retries_batch_with_passage_prefix() -> None:
    class OllamaFallbackGateway:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def post_json(self, **kwargs):
            self.calls.append(kwargs)
            texts = kwargs["payload"]["input"]
            if not texts[0].startswith("passage: "):
                raise AsyncCallRetryExhaustedError(
                    "embedding failed to encode response: "
                    "json: unsupported value: NaN"
                )
            return {"embeddings": [[1.0, 0.0] for _ in texts]}

    gateway = OllamaFallbackGateway()
    client = AsyncEmbeddingClient(
        config=AsyncEmbeddingConfig(
            provider="ollama",
            base_url="http://localhost:11434",
            model_name="bge-m3:latest",
        ),
        gateway=gateway,
    )

    vectors = await client.embed_texts(["def keyword_score(): pass"])

    assert vectors == [[1.0, 0.0]]
    assert gateway.calls[0]["payload"]["input"] == ["def keyword_score(): pass"]
    assert gateway.calls[1]["payload"]["input"] == ["passage: def keyword_score(): pass"]


@pytest.mark.asyncio
async def test_ollama_502_retries_batch_with_passage_prefix() -> None:
    class OllamaFallbackGateway:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def post_json(self, **kwargs):
            self.calls.append(kwargs)
            texts = kwargs["payload"]["input"]
            if not texts[0].startswith("passage: "):
                raise AsyncCallRetryExhaustedError(
                    "embedding failed: Server error '502 Bad Gateway'"
                )
            return {"embeddings": [[1.0, 0.0] for _ in texts]}

    gateway = OllamaFallbackGateway()
    client = AsyncEmbeddingClient(
        config=AsyncEmbeddingConfig(
            provider="ollama",
            base_url="http://localhost:11434",
            model_name="bge-m3:latest",
        ),
        gateway=gateway,
    )

    vectors = await client.embed_texts(["class SearchService: pass"])

    assert vectors == [[1.0, 0.0]]
    assert len(gateway.calls) == 2
    assert gateway.calls[1]["payload"]["input"] == [
        "passage: class SearchService: pass"
    ]
