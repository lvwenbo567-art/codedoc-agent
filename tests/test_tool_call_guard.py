from __future__ import annotations

from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from langgraph_agent.tool_call_guard import (
    build_tool_call_signature,
    evaluate_tool_calls,
)


def test_signature_ignores_argument_order() -> None:
    first = build_tool_call_signature(
        tool_name="search_code",
        arguments={"query": "x", "top_k": 5},
    )
    second = build_tool_call_signature(
        tool_name="search_code",
        arguments={"top_k": 5, "query": "x"},
    )

    assert first == second


def test_signature_changes_when_tool_name_changes() -> None:
    first = build_tool_call_signature(
        tool_name="search_code",
        arguments={"query": "x"},
    )
    second = build_tool_call_signature(
        tool_name="search_documents",
        arguments={"query": "x"},
    )

    assert first != second


def test_invalid_tool_call_is_blocked() -> None:
    result = evaluate_tool_calls(
        state={"tool_call_count": 0, "max_tool_calls": 10},
        tool_calls=[
            {"id": "call_1", "name": "delete_project", "args": {}},
        ],
        allowed_tool_names={"search_code"},
    )

    assert result.allowed is False
    assert result.stop_reason == "invalid_tool_call"


def test_tool_call_limit_is_blocked() -> None:
    result = evaluate_tool_calls(
        state={"tool_call_count": 2, "max_tool_calls": 2},
        tool_calls=[
            {"id": "call_1", "name": "search_code", "args": {}},
        ],
        allowed_tool_names={"search_code"},
    )

    assert result.allowed is False
    assert result.stop_reason == "tool_call_limit"


def test_first_and_second_identical_calls_are_allowed() -> None:
    first = evaluate_tool_calls(
        state={
            "tool_call_count": 0,
            "max_tool_calls": 10,
            "max_identical_tool_calls": 2,
        },
        tool_calls=[
            {
                "id": "call_1",
                "name": "search_code",
                "args": {"query": "RerankClient"},
            },
        ],
        allowed_tool_names={"search_code"},
    )
    second = evaluate_tool_calls(
        state={
            "tool_call_count": 1,
            "max_tool_calls": 10,
            "max_identical_tool_calls": 2,
            "tool_call_history": first.history_items,
        },
        tool_calls=[
            {
                "id": "call_2",
                "name": "search_code",
                "args": {"query": "RerankClient"},
            },
        ],
        allowed_tool_names={"search_code"},
    )

    assert first.allowed is True
    assert second.allowed is True
    assert second.history_items[0]["repeat_index"] == 2


def test_third_identical_call_is_blocked() -> None:
    first = evaluate_tool_calls(
        state={
            "tool_call_count": 0,
            "max_tool_calls": 10,
            "max_identical_tool_calls": 2,
        },
        tool_calls=[
            {
                "id": "call_1",
                "name": "search_code",
                "args": {"query": "RerankClient"},
            },
            {
                "id": "call_2",
                "name": "search_code",
                "args": {"query": "RerankClient"},
            },
        ],
        allowed_tool_names={"search_code"},
    )
    third = evaluate_tool_calls(
        state={
            "tool_call_count": 2,
            "max_tool_calls": 10,
            "max_identical_tool_calls": 2,
            "tool_call_history": first.history_items,
        },
        tool_calls=[
            {
                "id": "call_3",
                "name": "search_code",
                "args": {"query": "RerankClient"},
            },
        ],
        allowed_tool_names={"search_code"},
    )

    assert third.allowed is False
    assert third.stop_reason == "repeated_tool_call"


def test_same_ai_message_duplicate_obeys_repeat_limit() -> None:
    result = evaluate_tool_calls(
        state={
            "tool_call_count": 0,
            "max_tool_calls": 10,
            "max_identical_tool_calls": 1,
        },
        tool_calls=[
            {"id": "call_1", "name": "search_code", "args": {"query": "x"}},
            {"id": "call_2", "name": "search_code", "args": {"query": "x"}},
        ],
        allowed_tool_names={"search_code"},
    )

    assert result.allowed is False
    assert result.stop_reason == "repeated_tool_call"
