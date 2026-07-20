from dataclasses import dataclass
import re


@dataclass
class ProcessResult:
    """
    保存文本处理结果，方便测试和后续扩展。
    """

    original_text: str
    cleaned_text: str
    length: int
    truncated: bool


class TextProcessor:
    """
    文本处理器，负责清洗空白字符并按最大长度截断文本。
    """

    def __init__(self, max_length: int = 1000) -> None:
        """
        初始化文本处理器，并校验最大长度参数。
        """
        if max_length <= 0:
            raise ValueError("max_length 必须大于 0")

        self.max_length = max_length

    def clean_text(self, text: str) -> str:
        """
        去掉首尾空白，并把连续空白字符压缩成一个空格。
        """
        return re.sub(r"\s+", " ", text.strip())

    def process(self, text: str) -> ProcessResult:
        """
        清洗文本后按 max_length 截断，并返回结构化处理结果。
        """
        cleaned_text = self.clean_text(text)
        truncated = len(cleaned_text) > self.max_length

        if truncated:
            cleaned_text = cleaned_text[: self.max_length]

        return ProcessResult(
            original_text=text,
            cleaned_text=cleaned_text,
            length=len(cleaned_text),
            truncated=truncated,
        )
