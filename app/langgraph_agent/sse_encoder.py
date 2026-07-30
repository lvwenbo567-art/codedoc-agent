from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SSEEvent:
    '''
    这个类表示一个 SSE 事件。
    event
    事件类型，比如 token、node_update、interrupt、completed、error

    data
    事件数据，可以是 dict，也可以是字符串

    event_id
    可选事件 ID，用于前端断线重连时定位事件

    retry
    可选重试间隔，告诉浏览器断开后多久重连
    '''
    event: str
    data: Any
    event_id: str | None = None
    retry: int | None = None


def _encode_data_lines(data: Any) -> list[str]:
    #这个函数负责生成 SSE 的 data: 行。
    if isinstance(data, str):
        text = data
    else:
        text = json.dumps(
            data,
            ensure_ascii=False,
            default=str,
        )

    lines = text.splitlines()#SSE 的 data 支持多行。

    if not lines:
        return ["data:"]

    return [f"data: {line}" for line in lines]


def encode_sse_event(event: SSEEvent) -> str:
    #这个函数把一个 SSEEvent 编码成完整 SSE 字符串。
    lines: list[str] = []

    if event.event_id is not None:
        lines.append(f"id: {event.event_id}")

    lines.append(f"event: {event.event}")

    if event.retry is not None:
        lines.append(f"retry: {event.retry}")

    lines.extend(_encode_data_lines(event.data))
    lines.append("")
    lines.append("")

    return "\n".join(lines)


def encode_sse_comment(comment: str) -> str:
    #这个函数生成 SSE 注释。
    safe_comment = comment.replace("\r", " ").replace("\n", " ")

    return f": {safe_comment}\n\n"
