from __future__ import annotations

import json
from pathlib import Path

from evaluation.agent_eval_schema import AgentEvalCase


class AgentEvalDatasetError(ValueError):
    """
    评测集格式错误。
    """


def load_agent_eval_cases(dataset_path: str) -> list[AgentEvalCase]:
    """
    从 JSONL 文件加载 Agent 评测样例。

    每一行是一条 JSON；空行和 # 注释行会被跳过。
    """
    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"评测集不存在：{path}")

    if not path.is_file():
        raise AgentEvalDatasetError(f"评测集路径不是文件：{path}")

    cases: list[AgentEvalCase] = []
    seen_case_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            try:
                raw_case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AgentEvalDatasetError(
                    f"第 {line_number} 行不是有效 JSON：{exc}"
                ) from exc

            try:
                case = AgentEvalCase.model_validate(raw_case)
            except Exception as exc:
                raise AgentEvalDatasetError(
                    f"第 {line_number} 行评测数据非法：{exc}"
                ) from exc

            if case.case_id in seen_case_ids:
                raise AgentEvalDatasetError(
                    f"重复 case_id：{case.case_id}"
                )

            seen_case_ids.add(case.case_id)
            cases.append(case)

    if not cases:
        raise AgentEvalDatasetError("评测集不能为空")

    return cases
