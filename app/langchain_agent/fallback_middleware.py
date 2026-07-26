from __future__ import annotations

from langchain_agent.middleware_config import LangChainMiddlewareConfig
from langchain_agent.model_config import LangChainModelConfig
from langchain_agent.trace_recorder import AgentTraceRecorder


def build_fallback_model_config(
    *,
    primary_config: LangChainModelConfig,
    middleware_config: LangChainMiddlewareConfig,
) -> LangChainModelConfig | None:
    """
    根据 Middleware 配置构造备用模型配置。

    这里只负责配置生成，不主动调用模型。
    """
    if not middleware_config.fallback_enabled:
        return None

    if not middleware_config.fallback_model_name:
        return None

    return LangChainModelConfig(
        provider=primary_config.provider,
        model_name=middleware_config.fallback_model_name,
        base_url=(
            middleware_config.fallback_base_url
            or primary_config.base_url
        ),
        api_key=middleware_config.fallback_api_key,
        timeout_seconds=middleware_config.fallback_timeout_seconds,
        temperature=primary_config.temperature,
        max_tokens=primary_config.max_tokens,
        max_retries=primary_config.max_retries,
        structured_output_method=(
            primary_config.structured_output_method
        ),
    )


class CodeDocFallbackMiddleware:
    """
    备用模型降级标记组件。

    当前 Day33 先记录降级事件；真正复杂的多模型切换可在后续
    LangGraph 阶段继续增强。
    """

    def __init__(
        self,
        *,
        recorder: AgentTraceRecorder,
        fallback_config: LangChainModelConfig | None,
    ) -> None:
        self.recorder = recorder
        self.fallback_config = fallback_config

    def mark_if_enabled(self) -> None:
        if self.fallback_config is None:
            return

        self.recorder.mark_degraded(
            f"fallback model available: {self.fallback_config.model_name}"
        )
