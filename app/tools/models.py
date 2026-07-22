from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
'''
BaseModel	Pydantic所有数据模型的父类
ConfigDict	配置模型行为
Field	设置字段默认值、范围、描述
model_validator	校验多个字段之间的关系
'''

QueryStrategy = Literal[
    "original",
    "rewrite",
    "multi_query",
]


class StrictModel(BaseModel):
    """
    所有工具输入都禁止额外字段。

    防止模型生成未声明参数后，
    参数被静默忽略。
    """

    model_config = ConfigDict(
        extra="forbid",#如果传入了模型没有声明的字段，直接报错。
    )


class SearchCodeArgs(StrictModel):
    query: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "需要在代码中检索的问题。可以包含函数名、"
            "类名、方法名、文件名或自然语言描述。"
        ),
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="最终返回的代码结果数量。",
    )

    candidate_top_k: int = Field(
        default=20,
        ge=1,
        le=50,
        description=(
            "进入 Rerank 前的候选结果数量。"
            "必须大于或等于 top_k。"
        ),
    )

    query_strategy: QueryStrategy = Field(
        default="multi_query",
        description=(
            "查询策略。精确函数名可使用 original，"
            "模糊问题可使用 rewrite 或 multi_query。"
        ),
    )

    @model_validator(mode="after")#创建 SearchCodeArgs 对象时，还要执行这个方法进行整体校验。
    def validate_top_k(
        self,
    ) -> "SearchCodeArgs":
        if self.top_k > self.candidate_top_k:
            raise ValueError(
                "top_k 不能大于 candidate_top_k"
            )

        return self


class SearchDocumentsArgs(StrictModel):
    query: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "需要在 README、Markdown、PDF 文本或"
            "其他项目文档中检索的问题。"
        ),
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="最终返回的文档结果数量。",
    )

    candidate_top_k: int = Field(
        default=20,
        ge=1,
        le=50,
        description=(
            "进入 Rerank 前的候选结果数量。"
            "必须大于或等于 top_k。"
        ),
    )

    query_strategy: QueryStrategy = Field(
        default="multi_query",
        description=(
            "查询策略，可选 original、rewrite "
            "或 multi_query。"
        ),
    )

    @model_validator(mode="after")
    def validate_top_k(
        self,
    ) -> "SearchDocumentsArgs":
        if self.top_k > self.candidate_top_k:
            raise ValueError(
                "top_k 不能大于 candidate_top_k"
            )

        return self


class GetProjectStructureArgs(StrictModel):
    max_depth: int = Field(
        default=4,
        ge=1,
        le=8,
        description="返回目录树的最大深度。",
    )

    max_entries: int = Field(
        default=300,
        ge=10,
        le=1000,
        description=(
            "最多返回多少个文件和目录，"
            "避免结果过大。"
        ),
    )

    include_files: bool = Field(
        default=True,
        description=(
            "是否包含文件。为 false 时只返回目录。"
        ),
    )

    include_hidden: bool = Field(
        default=False,
        description=(
            "是否包含以点开头的隐藏文件和目录。"
        ),
    )


class ToolResult(StrictModel):
    """
    所有 Tool 统一返回此结构。
    """

    success: bool
    tool_name: str

    data: Any | None = None

    error_code: str | None = None
    error_message: str | None = None

    duration_ms: float = Field(
        default=0.0,
        ge=0,
    )
