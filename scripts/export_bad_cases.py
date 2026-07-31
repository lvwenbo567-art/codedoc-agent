from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
sys.path.append(str(APP_DIR))

from evaluation.agent_eval_report import export_failed_eval_results_as_jsonl
from evaluation.agent_eval_schema import AgentEvalReport
from repositories.feedback_repository import AgentFeedbackRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Day40 bad cases as Agent Eval JSONL."
    )
    parser.add_argument(
        "--source",
        choices=["eval-report", "feedback-db"],
        default="eval-report",
        help=(
            "eval-report: 从 Agent 评测报告中导出失败 Case；"
            "feedback-db: 从用户反馈数据库中导出已 promote 的 Bad Case。"
        ),
    )
    parser.add_argument(
        "--report",
        default="outputs/day40_agent_eval_report.json",
        help="source=eval-report 时使用的评测报告路径。",
    )
    parser.add_argument("--db-path", default="data/agent_feedback.db")
    parser.add_argument(
        "--output",
        default="data/evaluation/bad_cases.jsonl",
    )
    parser.add_argument("--project-id", type=int, default=None)
    return parser.parse_args()


def export_from_eval_report(
    *,
    report_path: str,
    output_path: str,
) -> dict:
    path = Path(report_path)

    if not path.exists():
        raise FileNotFoundError(
            "评测报告不存在，请先运行："
            "python scripts/run_day40_agent_eval.py"
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    report = AgentEvalReport.model_validate(payload)

    return export_failed_eval_results_as_jsonl(
        report=report,
        output_path=output_path,
    )


def export_from_feedback_db(
    *,
    db_path: str,
    output_path: str,
    project_id: int | None,
) -> dict:
    repository = AgentFeedbackRepository(db_path=db_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    bad_cases = repository.list_bad_cases(
        project_id=project_id,
        limit=1000,
        offset=0,
    )

    with output.open("w", encoding="utf-8") as file:
        for item in bad_cases:
            case = {
                "case_id": item["case_id"],
                "name": item["name"],
                "query": item["query"],
                "project_id": item["project_id"],
                "expected_tool_names": item["expected_tool_names"],
                "forbidden_tool_names": item["forbidden_tool_names"],
                "required_answer_terms": item["required_answer_terms"],
                "accepted_stop_reasons": item["accepted_stop_reasons"],
                "tags": ["bad_case", "feedback"],
                "notes": item["notes"],
            }
            file.write(json.dumps(case, ensure_ascii=False) + "\n")

    return {
        "output_path": str(output),
        "bad_case_count": len(bad_cases),
        "source": "feedback-db",
    }


def main() -> None:
    args = parse_args()

    if args.source == "eval-report":
        result = export_from_eval_report(
            report_path=args.report,
            output_path=args.output,
        )
    else:
        result = export_from_feedback_db(
            db_path=args.db_path,
            output_path=args.output,
            project_id=args.project_id,
        )

    print(result)


if __name__ == "__main__":
    main()
