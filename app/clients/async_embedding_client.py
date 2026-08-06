from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from runtime.async_call_policy import AsyncCallRetryExhaustedError
from runtime.async_http_gateway import AsyncHTTPGateway


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AsyncEmbeddingConfig:
    base_url: str
    model_name: str
    api_key: str | None = None
    batch_size: int = 32
    provider: str = "openai_compatible"

    def __post_init__(self) -> None:
        if self.provider not in {"ollama", "openai_compatible"}:
            raise ValueError(
                "Async Embedding provider 只支持 ollama 或 openai_compatible"
            )
        if not self.base_url.strip():
            raise ValueError("Embedding base_url 不能为空")
        if not self.model_name.strip():
            raise ValueError("Embedding model_name 不能为空")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")

    @property
    def endpoint(self) -> str:
        suffix = "/api/embed" if self.provider == "ollama" else "/embeddings"
        return self.base_url.rstrip("/") + suffix


class AsyncEmbeddingClient:
    """复用 AsyncHTTPGateway 的 Embedding 客户端。"""

    def __init__(self, *, config: AsyncEmbeddingConfig, gateway: AsyncHTTPGateway) -> None:
        self.config = config
        self.gateway = gateway

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def _request_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self.gateway.post_json(
            url=self.config.endpoint,
            payload={"model": self.config.model_name, "input": texts},
            headers=self._headers(),
            operation_name="embedding",
        )
        if self.config.provider == "ollama":
            raw_items = response.get("embeddings")
            ordered_items = (
                [{"embedding": item} for item in raw_items]
                if isinstance(raw_items, list)
                else None
            )
        else:
            raw_items = response.get("data")
            ordered_items = (
                sorted(raw_items, key=lambda item: int(item.get("index", 0)))
                if isinstance(raw_items, list)
                else None
            )
        if not isinstance(ordered_items, list):
            raise ValueError("Embedding response does not contain vectors")
        if len(ordered_items) != len(texts):
            raise ValueError("Embedding vector count does not match input count")

        vectors: list[list[float]] = []
        for item in ordered_items:
            vector = item.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise ValueError("Embedding response contains an invalid vector")
            converted = [float(value) for value in vector]
            if not all(math.isfinite(value) for value in converted):
                raise ValueError("Embedding vector contains NaN or Inf")
            vectors.append(converted)
        return vectors

    def _should_retry_ollama_batch_with_passage(
        self,
        error: Exception,
    ) -> bool:
        if self.config.provider != "ollama":
            return False
        message = str(error).lower()
        return (
            "unsupported value: nan" in message
            or "failed to encode response" in message
            or "502 bad gateway" in message
            or "status code: 502" in message
        )

    @staticmethod
    def _build_ollama_safe_embedding_text(text: str) -> str:
        if text.startswith("passage: "):
            return text
        return f"passage: {text}"

    async def embed_texts(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
    ) -> list[list[float]]:
        effective_batch_size = (
            self.config.batch_size
            if batch_size is None
            else batch_size
        )
        if effective_batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")

        normalized = [text.strip() for text in texts]
        if not normalized:
            return []
        if any(not text for text in normalized):
            raise ValueError("Embedding 输入不能包含空文本")

        vectors: list[list[float]] = []
        vector_dimension: int | None = None

        for start in range(0, len(normalized), effective_batch_size):
            batch = normalized[start:start + effective_batch_size]

            if self.config.provider == "ollama":
                try:
                    batch_vectors = await self._request_batch(batch)
                except AsyncCallRetryExhaustedError as exc:
                    if not self._should_retry_ollama_batch_with_passage(exc):
                        raise

                    safe_batch = [
                        self._build_ollama_safe_embedding_text(text)
                        for text in batch
                    ]
                    logger.warning(
                        "Ollama Embedding 第 %s 批（起始 Chunk=%s）返回 NaN 特征或 502，"
                        "使用 passage 前缀重试一次",
                        start // effective_batch_size + 1,
                        start,
                    )
                    batch_vectors = await self._request_batch(safe_batch)

                for converted in batch_vectors:
                    if vector_dimension is None:
                        vector_dimension = len(converted)
                    elif len(converted) != vector_dimension:
                        raise ValueError(
                            "Embedding returned inconsistent vector dimensions: "
                            f"{len(converted)} != {vector_dimension}"
                        )
                    vectors.append(converted)
                continue

            payload = {
                "model": self.config.model_name,
                "input": batch,
            }
            response = await self.gateway.post_json(
                url=self.config.endpoint, payload=payload, headers=self._headers(), operation_name="embedding"
            )
            if self.config.provider == "ollama":
                raw_items = response.get("embeddings")
                ordered_items = [{"embedding": item} for item in raw_items] if isinstance(raw_items, list) else None
            else:
                raw_items = response.get("data")
                ordered_items = sorted(raw_items, key=lambda item: int(item.get("index", 0))) if isinstance(raw_items, list) else None
            if not isinstance(ordered_items, list):
                raise ValueError("Embedding 响应缺少向量列表")
            for item in ordered_items:
                vector = item.get("embedding")
                if not isinstance(vector, list) or not vector:
                    raise ValueError("Embedding 响应缺少向量")
                converted = [float(value) for value in vector]
                if not all(math.isfinite(value) for value in converted):
                    raise ValueError("Embedding 向量不能包含 NaN 或 Inf")

                if vector_dimension is None:
                    vector_dimension = len(converted)
                elif len(converted) != vector_dimension:
                    raise ValueError(
                        "Embedding 返回的向量维度不一致："
                        f"{len(converted)} != {vector_dimension}"
                    )

                vectors.append(converted)
        if len(vectors) != len(normalized):
            raise ValueError("Embedding 返回数量与输入数量不一致")
        return vectors
