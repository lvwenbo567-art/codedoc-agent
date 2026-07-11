import hashlib
import math
import re
from typing import List

from config import DEFAULT_EMBEDDING_DIMENSION, DEFAULT_EMBEDDING_MODEL


class EmbeddingClient:
    """
    Embedding 客户端。

    当前使用本地哈希算法生成确定性向量，用于打通向量 RAG 链路。
    后续可以替换为 Ollama、OpenAI-compatible API 或其他 Embedding 服务。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    ):
        if dimension <= 0:
            raise ValueError("dimension 必须大于 0")

        self.model_name = model_name
        self.dimension = dimension

    def tokenize(self, text: str) -> List[str]:
        """
        将文本切分为简单 token。

        英文、数字和下划线组成完整 token；
        中文暂时按单个汉字切分。
        """
        if not text or not text.strip():
            return []

        return re.findall(
            r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]",
            text.lower(),
        )

    def embed_text(self, text: str) -> List[float]:
        """
        为单条文本生成向量。
        """
        tokens = self.tokenize(text)

        if not tokens:
            raise ValueError("待向量化文本不能为空")

        vector = [0.0] * self.dimension

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()

            # 使用哈希结果确定向量中的位置。
            index = int.from_bytes(digest[:4], byteorder="big") % self.dimension

            # 使用哈希结果确定加 1 还是减 1。
            sign = 1.0 if digest[4] % 2 == 0 else -1.0

            vector[index] += sign

        return self.normalize(vector)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文本向量。
        """
        return [self.embed_text(text) for text in texts]

    def normalize(self, vector: List[float]) -> List[float]:
        """
        对向量进行 L2 归一化。
        """
        norm = math.sqrt(sum(value * value for value in vector))

        if norm == 0:
            return vector

        return [value / norm for value in vector]