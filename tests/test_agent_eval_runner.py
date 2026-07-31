from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from evaluation.agent_eval_report import (
    export_failed_eval_results_as_jsonl,
    save_agent_eval_report,
)
from evaluation.agent_eval_runner import (
    extract_tool_names_from_output,
    run_agent_evaluation,
)
from evaluation.agent_eval_schema import AgentEvalCase, AgentEvalObservation


class FakeExecutor:
    def __init__(self, observations: list[AgentEvalObservation]) -> None:
        self.observations = observations
        self.index = 0

    async def execute(self, case: AgentEvalCase) -> AgentEvalObservation:
        observation = self.observations[self.index]
        self.index += 1
        return observation


def test_extract_tool_names_from_tool_history() -> None:
    output = {
        "tool_call_history": [
            {"tool_name": "search_code"},
            {"tool_name": "read_file_range"},
        ]
    }

    assert extract_tool_names_from_output(output) == [
        "search_code",
        "read_file_range",
    ]


def test_extract_tool_names_from_message_trace_fallback() -> None:
    output = {
        "message_trace": [
            {
                "tool_calls": [
                    {"name": "get_project_structure"},
                ]
            }
        ]
    }

    assert extract_tool_names_from_output(output) == [
        "get_project_structure"
    ]


@pytest.mark.asyncio
async def test_run_agent_evaluation_builds_report_and_summary(
    tmp_path: Path,
) -> None:
    cases = [
        AgentEvalCase(
            case_id="ok",
            name="OK",
            query="q",
            expected_tool_names=["search_code"],
            required_answer_terms=["keyword_score"],
        ),
        AgentEvalCase(
            case_id="bad",
            name="Bad",
            query="q",
            expected_tool_names=["read_file_range"],
            max_latency_ms=5,
        ),
    ]
    executor = FakeExecutor(
        [
            AgentEvalObservation(
                status="completed",
                answer="keyword_score",
                tool_names=["search_code"],
                stop_reason="completed",
                latency_ms=10,
                raw_output={"query": "q", "project_id": 1},
            ),
            AgentEvalObservation(
                status="completed",
                answer="answer",
                tool_names=[],
                stop_reason="completed",
                latency_ms=20,
                raw_output={"query": "q2", "project_id": 1},
            ),
        ]
    )

    report = await run_agent_evaluation(
        cases=cases,
        executor=executor,
        dataset_path="eval.jsonl",
        model_provider="mock",
        model_name="mock-model",
    )

    assert report.summary.total_cases == 2
    assert report.summary.passed_cases == 1
    assert report.summary.p95_latency_ms == 20

    output = tmp_path / "report.json"
    saved = save_agent_eval_report(report=report, output_path=str(output))

    assert saved["case_count"] == 2
    assert output.exists()

    bad_case_output = tmp_path / "bad_cases.jsonl"
    exported = export_failed_eval_results_as_jsonl(
        report=report,
        output_path=str(bad_case_output),
    )

    assert exported["bad_case_count"] == 1
    assert "bad-bad" in bad_case_output.read_text(encoding="utf-8")
