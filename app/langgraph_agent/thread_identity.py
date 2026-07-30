from __future__ import annotations

import re

'''

它负责把用户传进来的 thread_id 变成项目内部安全、隔离的 effective_thread_id。

'''
THREAD_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$"
)


class InvalidThreadIdError(ValueError):
    """
    对外 thread_id 不合法时抛出的错误。
    """


def validate_public_thread_id(
    thread_id: str,
) -> str:
    """
    校验用户传入的 thread_id，只允许安全、稳定、适合做命名空间的字符。
    """
    normalized = thread_id.strip()

    if not normalized:
        raise InvalidThreadIdError(
            "thread_id 不能为空"
        )

    if not THREAD_ID_PATTERN.fullmatch(
        normalized
    ):
        raise InvalidThreadIdError(
            "thread_id 只能包含字母、数字、点、下划线和短横线，"
            "长度为 1～120"
        )

    return normalized


def build_effective_thread_id(
    *,
    project_id: int,
    thread_id: str,
) -> str:
    """
    生成内部使用的 thread_id，把 project_id 放进命名空间，避免不同项目串记忆。
    """
    if project_id <= 0:
        raise ValueError(
            "project_id 必须大于 0"
        )

    normalized_thread_id = validate_public_thread_id(
        thread_id
    )

    return (
        f"project:{project_id}:"
        f"thread:{normalized_thread_id}"
    )
