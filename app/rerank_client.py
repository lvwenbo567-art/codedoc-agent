import re
from dataclasses import dataclass
from typing import Any

from config import (
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_DEVICE,
    DEFAULT_RERANK_LOCAL_FILES_ONLY,
    DEFAULT_RERANK_MAX_LENGTH,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_PROVIDER,
)


class RerankServiceError(RuntimeError):
    """
    Rerank 模型加载或推理失败。
    """


_RERANK_CLIENT_CACHE: dict[tuple, "RerankClient"] = {}


@dataclass(frozen=True)
class RerankConfig:
    """
    Reranker 配置，描述精排模型的来源、模型名和推理参数。
    """

    provider: str = DEFAULT_RERANK_PROVIDER
    model_name_or_path: str = DEFAULT_RERANK_MODEL
    device: str = DEFAULT_RERANK_DEVICE
    batch_size: int = DEFAULT_RERANK_BATCH_SIZE
    max_length: int = DEFAULT_RERANK_MAX_LENGTH
    local_files_only: bool = DEFAULT_RERANK_LOCAL_FILES_ONLY

    def validate(self) -> None:
        """
        校验 Rerank Provider 和基础推理参数是否合法。
        """
        supported_providers = {
            "mock",
            "sentence_transformers",
        }

        if self.provider not in supported_providers:
            raise ValueError(f"不支持的 Rerank Provider：{self.provider}")

        if not self.model_name_or_path.strip():
            raise ValueError("model_name_or_path 不能为空")

        if self.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")

        if self.max_length <= 0:
            raise ValueError("max_length 必须大于 0")

    def cache_key(self) -> tuple:
        """
        生成 RerankClient 缓存键，相同配置复用同一个模型实例。
        """
        return (
            self.provider,
            self.model_name_or_path,
            self.device,
            self.batch_size,
            self.max_length,
            self.local_files_only,
        )


class RerankClient:
    """
    Reranker 客户端，支持 mock 精排和 Sentence Transformers CrossEncoder。
    """

    def __init__(
        self,
        config: RerankConfig | None = None,
        model: Any = None,
    ) -> None:
        """
        初始化 RerankClient；测试时可以注入假模型，避免真实加载大模型。
        """
        self.config = config or RerankConfig()
        self.config.validate()
        self.model = model

    def score(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        """
        对 query 和多个候选文档组成的文本对进行相关性打分。
        """
        if not query or not query.strip():
            raise ValueError("query 不能为空")

        if not documents:
            return []

        for index, document in enumerate(documents):
            if not isinstance(document, str):
                raise TypeError(f"documents[{index}] 必须是字符串")

            if not document.strip():
                raise ValueError(f"documents[{index}] 不能为空")

        if self.config.provider == "mock":
            return [
                self._mock_score(query=query, document=document)
                for document in documents
            ]

        return self._cross_encoder_score(
            query=query,
            documents=documents,
        )

    def _mock_score(
        self,
        query: str,
        document: str,
    ) -> float:
        """
        使用简单 token 重合率模拟 Rerank 分数，便于单元测试稳定运行。
        """
        query_tokens = set(self._tokenize(query))
        document_tokens = set(self._tokenize(document))

        if not query_tokens:
            return 0.0

        overlap = query_tokens & document_tokens

        return len(overlap) / len(query_tokens)

    def _cross_encoder_score(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        """
        使用真实 CrossEncoder 对 query-document 文本对进行联合评分。
        """
        model = self._get_model()
        pairs = [
            [query, document]
            for document in documents
        ]
        try:
            scores = model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )

        except Exception as exc:
            raise RerankServiceError(
                f"Rerank 模型推理失败：{exc}"
            ) from exc

        return [
            float(score)
            for score in scores
        ]

    def _get_model(self) -> Any:
        """
        延迟加载真实 CrossEncoder 模型，只有真实 provider 被调用时才导入依赖。
        """
        if self.model is not None:
            return self.model

        try:
            from sentence_transformers import CrossEncoder

        except ImportError as exc:
            raise RerankServiceError(
                "未安装 sentence-transformers，请执行："
                "pip install sentence-transformers"
            ) from exc

        try:
            self.model = CrossEncoder(
                self.config.model_name_or_path,
                device=self.config.device,
                max_length=self.config.max_length,
                local_files_only=self.config.local_files_only,
            )

        except Exception as exc:
            raise RerankServiceError(
                f"Rerank 模型加载失败：{exc}"
            ) from exc

        return self.model

    def _tokenize(
        self,
        text: str,
    ) -> list[str]:
        """
        将中英文混合文本切成简单 token，用于 mock rerank。
        """
        return re.findall(
            r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]",
            text.lower(),
        )


def get_cached_rerank_client(config: RerankConfig) -> RerankClient:
    """
    获取可复用的 RerankClient，避免真实 CrossEncoder 在多次请求中重复加载。
    """
    config.validate()
    key = config.cache_key()

    if key not in _RERANK_CLIENT_CACHE:
        _RERANK_CLIENT_CACHE[key] = RerankClient(config=config)

    return _RERANK_CLIENT_CACHE[key]


def clear_rerank_client_cache() -> None:
    """
    清空 RerankClient 缓存，主要用于单元测试或手动释放模型引用。
    """
    _RERANK_CLIENT_CACHE.clear()
