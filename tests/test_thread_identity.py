from __future__ import annotations

from pathlib import Path
import sys

import pytest


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from langgraph_agent.thread_identity import (
    InvalidThreadIdError,
    build_effective_thread_id,
    validate_public_thread_id,
)


def test_valid_thread_id_returns_normalized_value() -> None:
    assert validate_public_thread_id(" demo-1 ") == "demo-1"
    assert validate_public_thread_id("interview.session_1") == "interview.session_1"


@pytest.mark.parametrize(
    "thread_id",
    [
        "",
        "   ",
        "has space",
        "bad/slash",
        "bad:colon",
        "-starts-with-dash",
        "x" * 121,
    ],
)
def test_invalid_thread_id_is_rejected(thread_id: str) -> None:
    with pytest.raises(InvalidThreadIdError):
        validate_public_thread_id(thread_id)


def test_effective_thread_id_contains_project_namespace() -> None:
    assert (
        build_effective_thread_id(
            project_id=1,
            thread_id="demo-1",
        )
        == "project:1:thread:demo-1"
    )


def test_effective_thread_id_isolated_by_project_id() -> None:
    first = build_effective_thread_id(project_id=1, thread_id="same")
    second = build_effective_thread_id(project_id=2, thread_id="same")

    assert first != second


def test_same_project_and_thread_build_same_effective_thread_id() -> None:
    first = build_effective_thread_id(project_id=3, thread_id="thread-a")
    second = build_effective_thread_id(project_id=3, thread_id="thread-a")

    assert first == second


def test_invalid_project_id_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_effective_thread_id(project_id=0, thread_id="demo")
