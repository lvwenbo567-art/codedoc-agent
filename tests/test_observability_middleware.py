from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1] / "app"
    )
)


from langchain_agent.observability_middleware import CodeDocObservabilityMiddleware
from langchain_agent.trace_recorder import AgentTraceRecorder


def test_observability_records_model_and_tool_call():
    recorder = AgentTraceRecorder(
        run_id="run-observe",
        trace_id="trace-observe",
    )
    middleware = CodeDocObservabilityMiddleware(
        recorder=recorder,
    )

    model_result = middleware.record_model_call(
        message_count=2,
        available_tool_count=5,
        func=lambda: {
            "ok": True,
        },
    )
    tool_result = middleware.record_tool_call(
        tool_call_id="call-1",
        tool_name="search_code",
        arguments={
            "query": "keyword_score",
        },
        func=lambda: {
            "result_count": 1,
        },
    )

    trace = recorder.snapshot()

    assert model_result["ok"] is True
    assert tool_result["result_count"] == 1
    assert len(trace.model_calls) == 1
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].tool_name == "search_code"
