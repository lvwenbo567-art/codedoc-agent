from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DEFAULT_READ_FILE_MAX_CHARS = 20000
MIN_USEFUL_READ_FILE_MAX_CHARS = 1000
DEFAULT_SYMBOL_MAX_CONTENT_CHARS = 12000


class StrictToolArgs(BaseModel):
    """
    确定性代码工具的参数基类。

    禁止模型生成未声明字段，防止参数被静默忽略。
    """

    model_config = ConfigDict(
        extra="forbid",
    )


class ReadFileRangeArgs(StrictToolArgs):
    source_path: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "相对于项目根目录的文本文件路径，"
            "例如 app/services/rag_service.py。"
        ),
    )

    start_line: int = Field(
        default=1,
        ge=1,
        description="开始行号，从 1 开始。",
    )

    end_line: int = Field(
        ge=1,
        description="结束行号，包含该行。",
    )

    max_chars: int = Field(
        default=DEFAULT_READ_FILE_MAX_CHARS,
        ge=100,
        le=50000,
        description=(
            "工具最多返回的字符数量，"
            "防止单次 Tool Result 过大。"
        ),
    )

    @field_validator("max_chars", mode="before")
    @classmethod
    def normalize_max_chars(cls, value: object) -> object:
        """
        兼容真实模型工具调用时传入 null 或过小 max_chars 的情况。
        """
        if value is None:
            return DEFAULT_READ_FILE_MAX_CHARS

        try:
            int_value = int(value)
        except (TypeError, ValueError):
            return value

        if int_value < MIN_USEFUL_READ_FILE_MAX_CHARS:
            return DEFAULT_READ_FILE_MAX_CHARS

        return value

    @model_validator(mode="after")
    def validate_line_range(self) -> "ReadFileRangeArgs":
        if self.end_line < self.start_line:
            raise ValueError("end_line 不能小于 start_line")

        if self.end_line - self.start_line + 1 > 300:
            raise ValueError("单次最多读取 300 行")

        return self


class GetSymbolDefinitionArgs(StrictToolArgs):
    symbol_name: str = Field(
        min_length=1,
        max_length=300,
        description=(
            "需要查找的函数、类或方法名称。"
            "可以是 score，也可以是 RerankClient.score。"
        ),
    )

    source_path: str | None = Field(
        default=None,
        max_length=500,
        description="可选的源文件路径，用于缩小查找范围。",
    )

    exact_match: bool = Field(
        default=True,
        description="是否优先执行精确符号匹配。",
    )

    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="最多返回多少个符号定义。",
    )

    max_content_chars: int = Field(
        default=DEFAULT_SYMBOL_MAX_CONTENT_CHARS,
        ge=500,
        le=30000,
        description="每个符号定义最多返回多少字符。",
    )

    @field_validator("max_content_chars", mode="before")
    @classmethod
    def normalize_max_content_chars(cls, value: object) -> object:
        """
        兼容真实模型工具调用时传入 null 的情况。
        """
        if value is None:
            return DEFAULT_SYMBOL_MAX_CONTENT_CHARS

        return value
