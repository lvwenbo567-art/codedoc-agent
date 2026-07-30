from __future__ import annotations

from pathlib import Path
import asyncio
import sys
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessageChunk


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from langgraph_agent.sse_encoder import SSEEvent
from langgraph_agent.sse_stream_service import (
    STREAM_END,
    HumanReviewSSEService,
    _compact_update,
    _extract_token_text,
)


class FakeRequest:
    def __init__(self, disconnected: bool = False) -> None:
        self.disconnected = disconnected

    async def is_disconnected(self) -> bool:
        return self.disconnected


class FakeGraph:
    async def astream(self, *args, **kwargs):
        yield (
            "messages",
            (
                AIMessageChunk(content="你"),
                {"langgraph_node": "agent"},
            ),
        )
        yield ("updates", {"agent": {"execution_steps": ["model_call"]}})

    async def aget_state(self, config):
        return SimpleNamespace(
            values={
                "answer": "完成",
                "stop_reason": "completed",
                "completed": True,
            },
            tasks=[],
        )


class FakeService:
    def __init__(self) -> None:
        self.graph = FakeGraph()


def test_extract_token_text_from_message_chunk() -> None:
    assert _extract_token_text(AIMessageChunk(content="abc")) == "abc"


def test_compact_update_keeps_key_fields() -> None:
    compact = _compact_update(
        {
            "node": {
                "execution_steps": ["x"],
                "stop_reason": "running",
                "large": "ignored",
            }
        }
    )

    assert compact == {
        "node": {
            "execution_steps": ["x"],
            "stop_reason": "running",
        }
    }


@pytest.mark.asyncio
async def test_produce_graph_events_emits_token_update_completed() -> None:
    service = HumanReviewSSEService(agent_service=FakeService())
    queue: asyncio.Queue = asyncio.Queue()

    await service._produce_graph_events(
        graph_input={},
        config={},
        queue=queue,
    )

    events = []

    while True:
        item = await queue.get()

        if item is STREAM_END:
            break

        events.append(item)

    assert [event.event for event in events] == [
        "token",
        "node_update",
        "completed",
    ]


@pytest.mark.asyncio
async def test_stream_with_heartbeat(monkeypatch) -> None:
    service = HumanReviewSSEService(agent_service=FakeService())
    calls = {"count": 0}

    async def fake_wait_for(awaitable, timeout):
        if hasattr(awaitable, "close"):
            awaitable.close()

        calls["count"] += 1

        if calls["count"] == 1:
            raise TimeoutError()

        return STREAM_END

    async def producer(queue):
        return None

    monkeypatch.setattr(
        "langgraph_agent.sse_stream_service.asyncio.wait_for",
        fake_wait_for,
    )

    chunks = [
        chunk
        async for chunk in service._stream_with_heartbeat(
            producer=producer,
            request=FakeRequest(),
        )
    ]

    assert any(": heartbeat" in chunk for chunk in chunks)
    assert any("event: connected" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_stream_disconnect_cancels_producer() -> None:
    service = HumanReviewSSEService(agent_service=FakeService())
    cancelled = {"value": False}

    async def producer(queue):
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled["value"] = True
            raise

    chunks = [
        chunk
        async for chunk in service._stream_with_heartbeat(
            producer=producer,
            request=FakeRequest(disconnected=True),
        )
    ]

    assert any("event: connected" in chunk for chunk in chunks)
    assert cancelled["value"] is True
