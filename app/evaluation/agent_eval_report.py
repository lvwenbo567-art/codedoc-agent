from __future__ import annotations

import json
from pathlib import Path

from evaluation.agent_eval_schema import AgentEvalCaseResult, AgentEvalReport


def save_agent_eval_report(
    *,
    report: AgentEvalReport,
    output_path: str,
) -> dict:
    """
    保存 Agent Evaluation JSON 报告。
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")#把 Pydantic 模型转成普通 Python dict。

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "output_path": str(path),
        "case_count": report.summary.total_cases,
        "passed_cases": report.summary.passed_cases,
        "failed_cases": report.summary.failed_cases,
    }


def get_failed_eval_results(
    report: AgentEvalReport,
) -> list[AgentEvalCaseResult]:
    """
    从评测报告中提取失败 Case。
    """
    return [result for result in report.results if not result.passed]


def export_failed_eval_results_as_jsonl(
    *,
    report: AgentEvalReport,
    output_path: str,
) -> dict:
    """
    将失败 Case 导出为 JSONL，作为后续 Bad Case 回归集。
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    failed_results = get_failed_eval_results(report)

    with path.open("w", encoding="utf-8") as file:
        for result in failed_results:
            observation = result.observation
            raw_query = observation.raw_output.get("query")
            query = raw_query if isinstance(raw_query, str) else result.name

            item = {
                "case_id": f"bad-{result.case_id}",
                "name": f"Bad Case：{result.name}",
                "query": query,
                "project_id": observation.raw_output.get("project_id", 1),
                "expected_status": "completed",
                "expected_tool_names": result.expected_tools,
                "forbidden_tool_names": result.forbidden_tools_used,
                "required_answer_terms": [
                    term for term in result.missing_answer_terms
                ],
                "accepted_stop_reasons": ["completed"],
                "tags": ["bad_case", "auto_exported"],
                "notes": "由 Day40 Agent Evaluation 失败结果自动导出。",
            }
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {
        "output_path": str(path),
        "bad_case_count": len(failed_results),
    }
