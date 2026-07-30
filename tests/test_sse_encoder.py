from __future__ import annotations

from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from langgraph_agent.sse_encoder import SSEEvent, encode_sse_comment, encode_sse_event


def test_encode_json_event_keeps_chinese() -> None:
    text = encode_sse_event(
        SSEEvent(event="node_update", data={"message": "你好"})
    )

    assert "event: node_update" in text
    assert '"message": "你好"' in text
    assert text.endswith("\n\n")


def test_encode_multiline_string_data() -> None:
    text = encode_sse_event(SSEEvent(event="token", data="a\nb"))

    assert "data: a" in text
    assert "data: b" in text


def test_encode_event_id_and_retry() -> None:
    text = encode_sse_event(
        SSEEvent(event="connected", data={}, event_id="1", retry=3000)
    )

    assert "id: 1" in text
    assert "retry: 3000" in text


def test_encode_comment_sanitizes_newlines() -> None:
    assert encode_sse_comment("hello\nworld") == ": hello world\n\n"
