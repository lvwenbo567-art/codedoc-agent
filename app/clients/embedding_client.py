import hashlib
import math
import re
import time
from dataclasses import dataclass
from typing import List, Optional

import httpx

from config import (
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    DEFAULT_NORMALIZE_EMBEDDING,
    DEFAULT_EMBEDDING_MAX_RETRIES,
    DEFAULT_EMBEDDING_RETRY_BACKOFF_SECONDS,
)


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Embedding 模型调用配置。
    """

    provider: str = DEFAULT_EMBEDDING_PROVIDER
    model_name: str = DEFAULT_EMBEDDING_MODEL
    base_url: str = DEFAULT_EMBEDDING_BASE_URL
    api_key: str = DEFAULT_EMBEDDING_API_KEY
    timeout_seconds: float = DEFAULT_EMBEDDING_TIMEOUT_SECONDS
    mock_dimension: int = DEFAULT_EMBEDDING_DIMENSION
    normalize: bool = DEFAULT_NORMALIZE_EMBEDDING
    max_retries: int = DEFAULT_EMBEDDING_MAX_RETRIES
    retry_backoff_seconds: float = DEFAULT_EMBEDDING_RETRY_BACKOFF_SECONDS
    def validate(self) -> None:
        """
        校验 Embedding 配置是否合法，避免运行时才发现模型参数错误。
        """
        supported_providers = {
            "mock",
            "ollama",
            "openai_compatible",
        }

        if self.provider not in supported_providers:
            raise ValueError(f"不支持的 Embedding Provider：{self.provider}")

        if not self.model_name.strip():
            raise ValueError("Embedding 模型名称不能为空")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0")

        if self.provider == "mock" and self.mock_dimension <= 0:
            raise ValueError("mock_dimension 必须大于 0")

        if self.provider != "mock" and not self.base_url.strip():
            raise ValueError("真实 Embedding Provider 必须配置 base_url")

        if self.max_retries < 0:
            raise ValueError(
                "max_retries 不能小于 0"
            )

        if self.retry_backoff_seconds < 0:
            raise ValueError(
                "retry_backoff_seconds 不能小于 0"
            )


class EmbeddingClient:
    """
    统一 Embedding 客户端。

    支持：
    1. mock
    2. ollama
    3. openai_compatible
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        http_client: Optional[httpx.Client] = None,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        """
        初始化 Embedding 客户端，并保留旧版本 model_name/dimension 调用方式。
        """
        if config is None:
            config = EmbeddingConfig(
                provider=DEFAULT_EMBEDDING_PROVIDER,
                model_name=model_name or DEFAULT_EMBEDDING_MODEL,
                mock_dimension=(
                    dimension
                    if dimension is not None
                    else DEFAULT_EMBEDDING_DIMENSION
                ),
            )

        config.validate()

        self.config = config
        self.http_client = http_client
        self.model_name = config.model_name
        self.dimension = config.mock_dimension
        self.request_count = 0
        self.retry_count = 0

    def get_call_stats(self) -> dict:
        """
        返回当前客户端的 HTTP 请求次数和重试次数。
        """
        return {
            "request_count": self.request_count,
            "retry_count": self.retry_count,
        }

    def embed_text(
        self,
        text: str,
    ) -> List[float]:
        """
        为单条文本生成 Embedding 向量。
        """
        return self.embed_texts([text])[0]

    def embed_texts(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        为一批文本生成 Embedding 向量。
        """
        self._validate_texts(texts)

        if self.config.provider == "mock":
            vectors = [
                self._embed_mock(text)
                for text in texts
            ]
        elif self.config.provider == "ollama":
            vectors = self._embed_ollama(texts)
        else:
            vectors = self._embed_openai_compatible(texts)

        self._validate_vectors(
            vectors=vectors,
            expected_count=len(texts),
        )

        if self.config.normalize:
            vectors = [
                self.normalize(vector)
                for vector in vectors
            ]

        return vectors

    def _validate_texts(
        self,
        texts: List[str],
    ) -> None:
        """
        校验待向量化文本列表，确保每条文本都是非空字符串。
        """
        if not texts:
            raise ValueError("texts 不能为空")

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"texts[{index}] 必须是字符串")

            if not text.strip():
                raise ValueError(f"texts[{index}] 不能为空")

    def _validate_vectors(
        self,
        vectors: List[List[float]],
        expected_count: int,
    ) -> None:
        """
        校验模型返回的向量数量、维度和数值类型是否符合预期。
        """
        if len(vectors) != expected_count:
            raise ValueError("Embedding 返回数量与输入数量不一致")

        if not vectors:
            raise ValueError("Embedding 服务没有返回向量")

        dimension = len(vectors[0])

        if dimension <= 0:
            raise ValueError("Embedding 向量维度不能为 0")

        for vector in vectors:
            if len(vector) != dimension:
                raise ValueError("Embedding 返回的向量维度不一致")

            if not all(isinstance(value, (int, float)) for value in vector):
                raise ValueError("Embedding 向量包含非法值")

            if not all(math.isfinite(float(value)) for value in vector):
                raise ValueError("Embedding 向量包含 NaN 或 Infinity")

    def _embed_mock(
        self,
        text: str,
    ) -> List[float]:
        """
        生成确定性的本地 Mock 向量，便于无模型环境下测试。
        """
        tokens = self.tokenize(text)

        vector = [0.0] * self.config.mock_dimension

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()

            index = (
                int.from_bytes(digest[:4], byteorder="big")
                % self.config.mock_dimension
            )

            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        return vector

    def _embed_ollama(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        调用 Ollama 原生 /api/embed 接口生成向量。
        """
        url = f"{self.config.base_url.rstrip('/')}/api/embed"

        payload = {
            "model": self.config.model_name,
            "input": texts,
        }

        try:
            data = self._post_json(
                url=url,
                payload=payload,
            )
        except RuntimeError as exc:
            if not self._is_ollama_nan_response_error(exc):
                raise

            retry_payload = {
                "model": self.config.model_name,
                "input": [
                    self._build_ollama_safe_embedding_text(text)
                    for text in texts
                ],
            }

            data = self._post_json(
                url=url,
                payload=retry_payload,
            )

        embeddings = data.get("embeddings")

        if not isinstance(embeddings, list):
            raise ValueError("Ollama Embedding 返回中缺少 embeddings")

        return embeddings

    def _is_ollama_nan_response_error(
        self,
        exc: RuntimeError,
    ) -> bool:
        """
        判断 Ollama 是否因为返回 NaN 向量导致 JSON 编码失败。
        """
        message = str(exc)

        return (
            "unsupported value: NaN" in message
            or "failed to encode response" in message
        )

    def _build_ollama_safe_embedding_text(
        self,
        text: str,
    ) -> str:
        """
        为 Ollama Embedding 构造更稳定的输入文本。

        bge 系列模型通常接受 passage 前缀；当原始短代码摘要触发
        Ollama NaN 响应时，追加前缀可以避开模型的边界输入。
        """
        if text.startswith("passage: "):
            return text

        return f"passage: {text}"

    def _embed_openai_compatible(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        调用 OpenAI-compatible /v1/embeddings 接口生成向量。
        """
        url = f"{self.config.base_url.rstrip('/')}/embeddings"

        payload = {
            "model": self.config.model_name,
            "input": texts,
        }

        data = self._post_json(
            url=url,
            payload=payload,
        )

        items = data.get("data")

        if not isinstance(items, list):
            raise ValueError("Embedding 返回中缺少 data")

        try:
            sorted_items = sorted(
                items,
                key=lambda item: item["index"],
            )

            return [
                item["embedding"]
                for item in sorted_items
            ]
        except (KeyError, TypeError) as exc:
            raise ValueError("Embedding 返回结构不符合预期") from exc

    def _post_json(
        self,
        url: str,
        payload: dict,
    ) -> dict:
        """
        发送 JSON 请求，并对超时、连接失败、429 和 5xx 做有限重试。
        """
        headers = {
            "Content-Type": "application/json",
        }

        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        owns_client = self.http_client is None
        client = self.http_client or httpx.Client(
            timeout=self.config.timeout_seconds,
            trust_env=False,
        )

        try:
            for attempt in range(self.config.max_retries + 1):
                self.request_count += 1

                try:
                    response = client.post(
                        url,
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()
                    data = response.json()

                    if not isinstance(data, dict):
                        raise ValueError("Embedding 返回结果必须是对象")

                    return data

                except httpx.TimeoutException as exc:
                    if attempt >= self.config.max_retries:
                        raise TimeoutError(
                            "Embedding 模型请求超时，并且已达到最大重试次数"
                        ) from exc

                    self._wait_before_retry(attempt)

                except httpx.RequestError as exc:
                    if attempt >= self.config.max_retries:
                        raise ConnectionError(
                            f"无法连接 Embedding 模型服务，并且已达到最大重试次数：{exc}"
                        ) from exc

                    self._wait_before_retry(attempt)

                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code
                    retryable = status_code == 429 or status_code >= 500

                    if not retryable or attempt >= self.config.max_retries:
                        response_text = exc.response.text.strip()
                        if len(response_text) > 500:
                            response_text = response_text[:500] + "..."

                        detail = f"Embedding 模型服务返回状态码：{status_code}"
                        if response_text:
                            detail = f"{detail}；响应内容：{response_text}"

                        raise RuntimeError(detail) from exc

                    self._wait_before_retry(attempt)

                except ValueError as exc:
                    raise ValueError("Embedding 模型返回了非法 JSON") from exc

            raise RuntimeError("Embedding 请求执行失败")

        finally:
            if owns_client:
                client.close()

    def _wait_before_retry(
        self,
        attempt: int,
    ) -> None:
        """
        使用指数退避等待后重试，并累计重试次数。
        """
        self.retry_count += 1

        delay = self.config.retry_backoff_seconds * (2 ** attempt)

        if delay > 0:
            time.sleep(delay)

    def tokenize(
        self,
        text: str,
    ) -> List[str]:
        """
        Mock Provider 使用的简单分词器。
        """
        return re.findall(
            r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]",
            text.lower(),
        )

    def normalize(
        self,
        vector: List[float],
    ) -> List[float]:
        """
        对向量做 L2 归一化，让余弦相似度计算更稳定。
        """
        norm = math.sqrt(
            sum(value * value for value in vector)
        )

        if norm == 0:
            return vector

        return [
            value / norm
            for value in vector
        ]
