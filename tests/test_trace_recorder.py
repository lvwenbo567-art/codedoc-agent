from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1] / "app"
    )
)


from langchain_agent.trace_recorder import AgentTraceRecorder, utc_now


def test_trace_recorder():
    recorder = AgentTraceRecorder(
        run_id="run-test",
        trace_id="trace-test",
    )

    started = utc_now()
    completed = utc_now()

    recorder.add_model_call(
        started_at=started,
        completed_at=completed,
        duration_ms=12.5,
        message_count=2,
        available_tool_count=5,
        success=True,
    )
    recorder.add_message_trim(
        original_count=30,
        kept_count=18,
    )
    recorder.mark_degraded(
        "fallback model used"
    )
    recorder.finish(
        status="completed",
        stop_reason="completed",
    )

    trace = recorder.snapshot()

    assert trace.run_id == "run-test"
    assert trace.trace_id == "trace-test"
    assert trace.status == "completed"
    assert len(trace.model_calls) == 1
    assert len(trace.message_trims) == 1
    assert trace.degraded is True
