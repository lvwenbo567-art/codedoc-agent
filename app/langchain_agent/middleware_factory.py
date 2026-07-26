from __future__ import annotations

from dataclasses import dataclass

from langchain_agent.fallback_middleware import (
    CodeDocFallbackMiddleware,
    build_fallback_model_config,
)
from langchain_agent.message_window_middleware import CodeDocMessageWindowMiddleware
from langchain_agent.middleware_config import LangChainMiddlewareConfig
from langchain_agent.model_config import LangChainModelConfig
from langchain_agent.observability_middleware import CodeDocObservabilityMiddleware
from langchain_agent.trace_recorder import AgentTraceRecorder


class ModelCallLimitExceededError(RuntimeError):
    pass


class ToolCallLimitExceededError(RuntimeError):
    pass


@dataclass(slots=True)#表示该对象只能拥有已经声明的属性：
class CodeDocMiddlewareBundle:
    observability: CodeDocObservabilityMiddleware
    #用于记录 Agent 运行过程，例如：模型调用；工具调用；错误信息；耗时；降级状态；Agent Trace。
    
    message_window: CodeDocMessageWindowMiddleware

    fallback: CodeDocFallbackMiddleware


def build_agent_middleware(
    *,
    config: LangChainMiddlewareConfig,
    recorder: AgentTraceRecorder,
    primary_model_config: LangChainModelConfig | None = None,
) -> CodeDocMiddlewareBundle:
    """
    组装 Day33 可靠性组件。

    返回项目层 middleware bundle，由 AgentService 显式调用。
    """
    fallback_config = None

    if primary_model_config is not None:
        fallback_config = build_fallback_model_config(
            primary_config=primary_model_config,
            middleware_config=config,
        )

    return CodeDocMiddlewareBundle(
        observability=CodeDocObservabilityMiddleware(
            recorder=recorder,
        ),
        message_window=CodeDocMessageWindowMiddleware(
            max_messages=config.max_model_messages,
            recorder=recorder,
        ),
        fallback=CodeDocFallbackMiddleware(
            recorder=recorder,
            fallback_config=fallback_config,
        ),
    )
