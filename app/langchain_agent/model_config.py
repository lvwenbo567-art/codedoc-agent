from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
'''
BaseModel	定义可校验的数据模型
ConfigDict	配置 Pydantic 模型自身的行为
Field	为字段设置默认值和校验条件
SecretStr	保存密码、API Key 等敏感字符串
model_validator	对整个模型进行补充校验
'''

LangChainProvider = Literal[
    "mock",
    "openai_compatible",
]

StructuredOutputMethod = Literal[
    "function_calling",
    "json_schema",
]


class LangChainModelConfig(BaseModel):
    """
    LangChain ChatModel 配置。

    mock:
        不连接真实模型，主要用于接口、Message 构建和规则分类测试。

    openai_compatible:
        连接 Ollama、vLLM 或其他兼容 OpenAI Chat Completions 的模型服务。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    provider: LangChainProvider = "mock"
    model_name: str = "mock-langchain-model"
    base_url: str = "http://localhost:11434/v1"
    api_key: SecretStr = SecretStr("EMPTY")
    timeout_seconds: float = Field(default=60.0, gt=0)
    temperature: float = Field(default=0.1, ge=0, le=2)
    max_tokens: int = Field(default=1200, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    structured_output_method: StructuredOutputMethod = "function_calling"

    @model_validator(mode="after")
    def validate_config(self) -> "LangChainModelConfig":
        """
        校验模型配置，避免运行到模型调用阶段才发现基础参数错误。
        """
        if not self.model_name.strip():
            raise ValueError("model_name 不能为空")

        if self.provider == "openai_compatible" and not self.base_url.strip():
            raise ValueError("openai_compatible 模式必须配置 base_url")

        return self

    @classmethod
    def from_env(cls) -> "LangChainModelConfig":
        """
        从环境变量构建 LangChain 配置，方便在 mock、Ollama、vLLM 之间切换。
        """
        return cls(
            provider=os.getenv("LANGCHAIN_CHAT_PROVIDER", "mock"),
            model_name=os.getenv(
                "LANGCHAIN_CHAT_MODEL",
                "mock-langchain-model",
            ),
            base_url=os.getenv(
                "LANGCHAIN_CHAT_BASE_URL",
                "http://localhost:11434/v1",
            ),
            api_key=SecretStr(
                os.getenv(
                    "LANGCHAIN_CHAT_API_KEY",
                    "EMPTY",
                )
            ),
            timeout_seconds=float(
                os.getenv(
                    "LANGCHAIN_CHAT_TIMEOUT_SECONDS",
                    "60",
                )
            ),
            temperature=float(
                os.getenv(
                    "LANGCHAIN_CHAT_TEMPERATURE",
                    "0.1",
                )
            ),
            max_tokens=int(
                os.getenv(
                    "LANGCHAIN_CHAT_MAX_TOKENS",
                    "1200",
                )
            ),
            max_retries=int(
                os.getenv(
                    "LANGCHAIN_CHAT_MAX_RETRIES",
                    "2",
                )
            ),
            structured_output_method=os.getenv(
                "LANGCHAIN_STRUCTURED_OUTPUT_METHOD",
                "function_calling",
            ),
        )

    def safe_dict(self) -> dict:
        """
        返回可以通过 API 展示的配置，不泄露 api_key 明文。
        """
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "api_key_configured": self.api_key.get_secret_value()
            not in {
                "",
                "EMPTY",
            },
            "timeout_seconds": self.timeout_seconds,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
            "structured_output_method": self.structured_output_method,
        }

