from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1] / "app"
    )
)


from langchain_agent.middleware_config import LangChainMiddlewareConfig
from langchain_agent.middleware_factory import build_agent_middleware
from langchain_agent.model_config import LangChainModelConfig
from langchain_agent.trace_recorder import AgentTraceRecorder


def test_langchain_middleware_factory_builds_bundle():
    config = LangChainMiddlewareConfig(
        max_model_messages=8,
        fallback_enabled=True,
        fallback_model_name="backup-model",
    )
    recorder = AgentTraceRecorder(
        run_id="run-factory",
        trace_id="trace-factory",
    )

    bundle = build_agent_middleware(
        config=config,
        recorder=recorder,
        primary_model_config=LangChainModelConfig(
            provider="openai_compatible",
            model_name="primary-model",
            base_url="http://localhost:11434/v1",
        ),
    )

    assert bundle.message_window.max_messages == 8
    assert bundle.fallback.fallback_config is not None
    assert bundle.fallback.fallback_config.model_name == "backup-model"
