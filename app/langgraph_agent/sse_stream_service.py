from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi import Request
from langchain_core.messages import AIMessageChunk
from langgraph.types import Command

from langgraph_agent.human_review_schema import HumanReviewDecision
from langgraph_agent.human_review_service import (
    HumanReviewToolAgentService,
    extract_interrupts,
)
from langgraph_agent.sse_encoder import SSEEvent, encode_sse_comment, encode_sse_event
from langgraph_agent.thread_identity import build_effective_thread_id


STREAM_END = object()#这是一个特殊哨兵对象。 用来表示 producer 已经结束


def _extract_token_text(message_chunk: Any) -> str:
    #这个函数从模型流式 chunk 中提取文本。
    content = getattr(message_chunk, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text")

                if text:
                    parts.append(str(text))

        return "".join(parts)

    return ""


def _compact_update(update_data: Any) -> dict[str, Any]:
    '''这个函数负责压缩 LangGraph 的 node update。'''
    if not isinstance(update_data, dict):
        return {"update": str(update_data)}

    compact: dict[str, Any] = {}

    for node_name, node_update in update_data.items():
        if node_name == "__interrupt__":
            compact[node_name] = str(node_update)
            continue

        if not isinstance(node_update, dict):
            compact[node_name] = str(node_update)
            continue

        compact[node_name] = {
            key: node_update[key]
            for key in (
                "execution_steps",
                "stop_reason",
                "approval_status",
                "completed",
                "answer",
                "error_message",
            )
            if key in node_update
        }

    return compact


def _normalize_stream_part(part: Any) -> tuple[str | None, Any]:
    """
    兼容 LangGraph 不同版本的 stream 输出形态。
    """
    if isinstance(part, dict):
        event_type = part.get("event") or part.get("type")
        data = part.get("data", part)

        return str(event_type) if event_type else None, data

    if isinstance(part, tuple) and len(part) == 2:
        event_type, data = part

        return str(event_type), data

    return None, part


class HumanReviewSSEService:
    def __init__(
        self,
        *,
        agent_service: HumanReviewToolAgentService,
    ) -> None:
        self.agent_service = agent_service

    async def _produce_graph_events(
        self,
        *,
        graph_input: Any,
        config: dict[str, Any],
        queue: asyncio.Queue[Any],
    ) -> None:
        try:
            async for raw_part in self.agent_service.graph.astream(
                graph_input,
                config=config,
                stream_mode=["messages", "updates"],
                version="v2",
                durability="sync",
            ):
                event_type, data = _normalize_stream_part(raw_part)

                if event_type == "messages":
                    message_chunk: Any
                    metadata: dict[str, Any]

                    if isinstance(data, tuple) and len(data) == 2:
                        message_chunk, metadata = data
                    else:
                        message_chunk = data
                        metadata = {}

                    token = _extract_token_text(message_chunk)

                    if token:
                        await queue.put(
                            SSEEvent(
                                event="token",
                                data={
                                    "node": metadata.get("langgraph_node"),
                                    "text": token,
                                },
                            )
                        )

                elif event_type == "updates":
                    await queue.put(
                        SSEEvent(
                            event="node_update",
                            data=_compact_update(data),
                        )
                    )

            snapshot = await self.agent_service.graph.aget_state(config)
            values = getattr(snapshot, "values", None) or {}
            interrupts = extract_interrupts(snapshot)

            if interrupts:
                await queue.put(
                    SSEEvent(
                        event="interrupt",
                        data={
                            "status": "interrupted",
                            "interrupts": interrupts,
                        },
                    )
                )
            else:
                await queue.put(
                    SSEEvent(
                        event="completed",
                        data={
                            "answer": values.get("answer"),
                            "stop_reason": values.get("stop_reason"),
                            "completed": values.get("completed"),
                        },
                    )
                )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await queue.put(
                SSEEvent(
                    event="error",
                    data={
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            )
        finally:
            await queue.put(STREAM_END)

    async def _stream_with_heartbeat(
        self,
        *,
        producer: Callable[[asyncio.Queue[Any]], Awaitable[None]],
        #一个异步函数 它接收 queue 它负责往 queue 里放事件
        request: Request,
    ) -> AsyncIterator[str]:#意思是这个函数会异步不断 yield 字符串
        queue: asyncio.Queue[Any] = asyncio.Queue()
        producer_task = asyncio.create_task(producer(queue))
        await asyncio.sleep(0)#让出一次事件循环，让 producer_task 有机会真正启动

        yield encode_sse_event(
            SSEEvent(
                event="connected",
                data={"status": "connected"},
                retry=3000,
            )
        )

        try:
            while True:
                if await request.is_disconnected():
                    producer_task.cancel()
                    break#如果前端断开了连接，就取消后台 producer，然后退出循环。

                try:
                    item = await asyncio.wait_for(
                        queue.get(),
                        timeout=15,
                    )#等 queue 里出现一个事件 但最多等 15 秒
                except TimeoutError:
                    yield encode_sse_comment("heartbeat")#如果 15 秒没有事件，就发：: heartbeat
                                                         #这个不会触发业务事件，但能保持连接。
                    continue

                if item is STREAM_END:
                    break

                yield encode_sse_event(item)#编码然后 yield 给 FastAPI。 FastAPI 再推给前端。

        finally:
            if not producer_task.done():
                producer_task.cancel()

            try:
                await producer_task
            except asyncio.CancelledError:
                pass

    async def stream_start(
        self,
        *,
        query: str,
        project_id: int,
        thread_id: str,
        recursion_limit: int,
        request: Request,
    ) -> AsyncIterator[str]:
        effective_thread_id = build_effective_thread_id(
            project_id=project_id,
            thread_id=thread_id,
        )
        run_id = f"stream_{id(request)}"
        graph_input = self.agent_service.build_turn_input(
            query=query,
            project_id=project_id,
            thread_id=thread_id,
            effective_thread_id=effective_thread_id,
            run_id=run_id,
        )
        config = self.agent_service.build_config(
            project_id=project_id,
            thread_id=thread_id,
            run_id=run_id,
            recursion_limit=recursion_limit,
        )
        lock = self.agent_service.thread_lock_provider(effective_thread_id)

        async def producer(queue: asyncio.Queue[Any]) -> None:
            async with lock:
                await self._produce_graph_events(
                    graph_input=graph_input,
                    config=config,
                    queue=queue,
                )

        async for chunk in self._stream_with_heartbeat(
            producer=producer,
            request=request,
        ):
            yield chunk

    async def stream_resume(
        self,
        *,
        project_id: int,
        thread_id: str,
        decision: HumanReviewDecision,
        recursion_limit: int,
        request: Request,
    ) -> AsyncIterator[str]:
        effective_thread_id = build_effective_thread_id(
            project_id=project_id,
            thread_id=thread_id,
        )
        config = self.agent_service.build_config(
            project_id=project_id,
            thread_id=thread_id,
            run_id=f"stream_resume_{id(request)}",
            recursion_limit=recursion_limit,
        )
        lock = self.agent_service.thread_lock_provider(effective_thread_id)

        async def producer(queue: asyncio.Queue[Any]) -> None:
            async with lock:
                await self._produce_graph_events(
                    graph_input=Command(resume=decision.model_dump()),
                    config=config,
                    queue=queue,
                )

        async for chunk in self._stream_with_heartbeat(
            producer=producer,
            request=request,
        ):
            yield chunk
