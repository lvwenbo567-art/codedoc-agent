from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from evaluation.agent_eval_metrics import (
    evaluate_agent_case,
    summarize_agent_eval_results,
)
from evaluation.agent_eval_schema import AgentEvalCase, AgentEvalObservation


def make_observation(
    *,
    tools: list[str],
    answer: str = "keyword_score search.py completed",
    latency_ms: float = 10,
    stop_reason: str = "completed",
) -> AgentEvalObservation:
    return AgentEvalObservation(
        status="completed",
        answer=answer,
        tool_names=tools,
        stop_reason=stop_reason,
        latency_ms=latency_ms,
    )


def test_evaluate_agent_case_success() -> None:
    case = AgentEvalCase(
        case_id="ok",
        name="OK",
        query="q",
        expected_tool_names=["get_symbol_definition"],
        expected_first_tool="get_symbol_definition",
        required_answer_terms=["keyword_score", "search.py"],
        max_latency_ms=100,
    )
    result = evaluate_agent_case(
        case=case,
        observation=make_observation(tools=["get_symbol_definition"]),
    )

    assert result.task_success == 1.0
    assert result.tool_recall == 1.0
    assert result.tool_precision == 1.0
    assert result.tool_f1 == 1.0
    assert result.first_tool_accuracy == 1.0
    assert result.answer_term_coverage == 1.0


def test_evaluate_agent_case_missing_expected_tool() -> None:
    case = AgentEvalCase(
        case_id="missing-tool",
        name="Missing Tool",
        query="q",
        expected_tool_names=["search_code"],
    )
    result = evaluate_agent_case(
        case=case,
        observation=make_observation(tools=[]),
    )

    assert result.tool_recall == 0.0
    assert "search_code" in result.missing_tools
    assert result.task_success == 0.0


def test_evaluate_agent_case_extra_tool_lowers_precision() -> None:
    case = AgentEvalCase(
        case_id="extra-tool",
        name="Extra Tool",
        query="q",
        expected_tool_names=["search_code"],
    )
    result = evaluate_agent_case(
        case=case,
        observation=make_observation(
            tools=["search_code", "read_file_range"]
        ),
    )

    assert result.tool_recall == 1.0
    assert result.tool_precision == 0.5
    assert result.unexpected_tools == ["read_file_range"]


def test_evaluate_agent_case_forbidden_tool() -> None:
    case = AgentEvalCase(
        case_id="forbidden",
        name="Forbidden",
        query="q",
        forbidden_tool_names=["read_file_range"],
    )
    result = evaluate_agent_case(
        case=case,
        observation=make_observation(tools=["read_file_range"]),
    )

    assert result.forbidden_tool_safety == 0.0
    assert result.forbidden_tools_used == ["read_file_range"]
    assert result.task_success == 0.0


def test_evaluate_agent_case_no_expected_tool_but_actual_tool() -> None:
    case = AgentEvalCase(
        case_id="out-of-scope",
        name="Out Of Scope",
        query="q",
        expected_tool_names=[],
    )
    result = evaluate_agent_case(
        case=case,
        observation=make_observation(tools=["search_code"]),
    )

    assert result.tool_recall == 1.0
    assert result.tool_precision == 0.0
    assert result.tool_exact_match == 0.0


def test_evaluate_agent_case_first_tool_wrong() -> None:
    case = AgentEvalCase(
        case_id="first-tool",
        name="First Tool",
        query="q",
        expected_tool_names=["search_code"],
        expected_first_tool="search_code",
    )
    result = evaluate_agent_case(
        case=case,
        observation=make_observation(
            tools=["get_project_structure", "search_code"]
        ),
    )

    assert result.first_tool_accuracy == 0.0
    assert "第一次工具选择不符合预期" in result.failure_reasons


def test_evaluate_agent_case_missing_answer_terms() -> None:
    case = AgentEvalCase(
        case_id="terms",
        name="Terms",
        query="q",
        required_answer_terms=["keyword_score", "search.py"],
    )
    result = evaluate_agent_case(
        case=case,
        observation=make_observation(tools=[], answer="keyword_score"),
    )

    assert result.answer_term_coverage == 0.5
    assert result.missing_answer_terms == ["search.py"]


def test_evaluate_agent_case_latency_failure() -> None:
    case = AgentEvalCase(
        case_id="latency",
        name="Latency",
        query="q",
        max_latency_ms=5,
    )
    result = evaluate_agent_case(
        case=case,
        observation=make_observation(tools=[], latency_ms=10),
    )

    assert result.latency_pass == 0.0
    assert "执行延迟超过阈值" in result.failure_reasons


def test_summarize_agent_eval_results() -> None:
    ok_case = AgentEvalCase(case_id="ok", name="OK", query="q")
    bad_case = AgentEvalCase(
        case_id="bad",
        name="Bad",
        query="q",
        required_answer_terms=["missing"],
    )
    results = [
        evaluate_agent_case(
            case=ok_case,
            observation=make_observation(tools=[]),
        ),
        evaluate_agent_case(
            case=bad_case,
            observation=make_observation(tools=[], answer="nope"),
        ),
    ]

    summary = summarize_agent_eval_results(results)

    assert summary.total_cases == 2
    assert summary.passed_cases == 1
    assert summary.failed_cases == 1
    assert summary.task_success_rate == 0.5
    assert summary.p95_latency_ms == 10
