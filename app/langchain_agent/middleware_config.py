from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field, SecretStr

'''
Agent 运行时的安全配置表

'''
def _env_bool(
    name: str,
    default: bool,
) -> bool:
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    return raw_value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class LangChainMiddlewareConfig(BaseModel):
    """
    CodeDoc LangChain Agent 的可靠性配置。

    这里集中管理模型调用限制、工具调用限制、重试、消息窗口和备用模型。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    model_run_limit: int = Field(
        default=6,
        ge=1,
        le=20,
    )
    tool_run_limit: int = Field(
        default=10,
        ge=1,
        le=50,
    )
    model_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
    )
    tool_max_retries: int = Field(
        default=1,
        ge=0,
        le=3,
    )
    retry_initial_delay: float = Field(
        default=0.5,
        ge=0,
        le=10,
    )
    retry_backoff_factor: float = Field(
        default=2.0,
        ge=0,
        le=10,
    )
    retry_max_delay: float = Field(
        default=4.0,
        ge=0,
        le=60,
    )
    retry_jitter: bool = True
    '''
    加入随机抖动后，等待时间可能变为：

    请求A：0.83秒
    请求B：1.07秒
    请求C：1.22秒

    从而把重试请求分散开。
    '''
    max_model_messages: int = Field(
        default=18,
        ge=4,
        le=100,
    )
    fallback_enabled: bool = False
    fallback_model_name: str | None = None
    fallback_base_url: str | None = None
    fallback_api_key: SecretStr = SecretStr("EMPTY")
    fallback_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
    )

    @classmethod
    def from_env(cls) -> "LangChainMiddlewareConfig":
        return cls(
            model_run_limit=int(os.getenv("LANGCHAIN_MODEL_RUN_LIMIT", "6")),
            tool_run_limit=int(os.getenv("LANGCHAIN_TOOL_RUN_LIMIT", "10")),
            model_max_retries=int(os.getenv("LANGCHAIN_MODEL_MAX_RETRIES", "2")),
            tool_max_retries=int(os.getenv("LANGCHAIN_TOOL_MAX_RETRIES", "1")),
            retry_initial_delay=float(
                os.getenv("LANGCHAIN_RETRY_INITIAL_DELAY", "0.5")
            ),
            retry_backoff_factor=float(
                os.getenv("LANGCHAIN_RETRY_BACKOFF_FACTOR", "2.0")
            ),
            retry_max_delay=float(os.getenv("LANGCHAIN_RETRY_MAX_DELAY", "4.0")),
            retry_jitter=_env_bool("LANGCHAIN_RETRY_JITTER", True),
            max_model_messages=int(os.getenv("LANGCHAIN_MAX_MODEL_MESSAGES", "18")),
            fallback_enabled=_env_bool("LANGCHAIN_FALLBACK_ENABLED", False),
            fallback_model_name=os.getenv("LANGCHAIN_FALLBACK_MODEL"),
            fallback_base_url=os.getenv("LANGCHAIN_FALLBACK_BASE_URL"),
            fallback_api_key=SecretStr(
                os.getenv("LANGCHAIN_FALLBACK_API_KEY", "EMPTY")
            ),
            fallback_timeout_seconds=float(
                os.getenv("LANGCHAIN_FALLBACK_TIMEOUT_SECONDS", "60")
            ),
        )

    def safe_dict(self) -> dict:
        return {
            "model_run_limit": self.model_run_limit,
            "tool_run_limit": self.tool_run_limit,
            "model_max_retries": self.model_max_retries,
            "tool_max_retries": self.tool_max_retries,
            "retry_initial_delay": self.retry_initial_delay,
            "retry_backoff_factor": self.retry_backoff_factor,
            "retry_max_delay": self.retry_max_delay,
            "retry_jitter": self.retry_jitter,
            "max_model_messages": self.max_model_messages,
            "fallback_enabled": self.fallback_enabled,
            "fallback_model_name": self.fallback_model_name,
            "fallback_base_url": self.fallback_base_url,
            "fallback_api_key_configured": (
                self.fallback_api_key.get_secret_value()
                not in {
                    "",
                    "EMPTY",
                }
            ),
            "fallback_timeout_seconds": self.fallback_timeout_seconds,
        }
