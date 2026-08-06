from __future__ import annotations

from typing import Protocol


class TokenCounter(Protocol):
    """Token 计数接口，后续可替换为特定模型的真实 tokenizer。"""

    def count_text(self, text: str) -> int:
        """返回文本占用的 Token 数。"""


class ApproximateTokenCounter:
    """开发阶段的近似计数器：中文约 1 Token，其他字符约每 4 个 1 Token。"""

    def count_text(self, text: str) -> int:
        if not text:
            return 0

        chinese_count = sum("\u4e00" <= char <= "\u9fff" for char in text)
        other_count = len(text) - chinese_count
        return chinese_count + (other_count + 3) // 4


class CharacterTokenCounter:
    """测试专用计数器：每个字符视为一个 Token。"""

    def count_text(self, text: str) -> int:
        return len(text)
