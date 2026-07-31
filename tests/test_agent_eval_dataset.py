from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from evaluation.agent_eval_dataset import (
    AgentEvalDatasetError,
    load_agent_eval_cases,
)


def write_text(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_load_agent_eval_cases_supports_jsonl_comments_and_blank_lines(
    tmp_path: Path,
) -> None:
    dataset = write_text(
        tmp_path / "eval.jsonl",
        """
# comment

{"case_id":"case-1","name":"Case 1","query":"keyword_score 在哪？"}
{"case_id":"case-2","name":"Case 2","query":"项目结构是什么？","expected_tool_names":["get_project_structure"]}
""",
    )

    cases = load_agent_eval_cases(str(dataset))

    assert len(cases) == 2
    assert cases[0].case_id == "case-1"
    assert cases[1].expected_tool_names == ["get_project_structure"]


def test_load_agent_eval_cases_rejects_duplicate_case_id(
    tmp_path: Path,
) -> None:
    dataset = write_text(
        tmp_path / "eval.jsonl",
        """
{"case_id":"dup","name":"Case 1","query":"q1"}
{"case_id":"dup","name":"Case 2","query":"q2"}
""",
    )

    with pytest.raises(AgentEvalDatasetError, match="重复 case_id"):
        load_agent_eval_cases(str(dataset))


def test_load_agent_eval_cases_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    dataset = write_text(tmp_path / "eval.jsonl", "{bad json}\n")

    with pytest.raises(AgentEvalDatasetError, match="不是有效 JSON"):
        load_agent_eval_cases(str(dataset))


def test_load_agent_eval_cases_rejects_missing_query(
    tmp_path: Path,
) -> None:
    dataset = write_text(
        tmp_path / "eval.jsonl",
        '{"case_id":"missing-query","name":"Missing Query"}\n',
    )

    with pytest.raises(AgentEvalDatasetError, match="评测数据非法"):
        load_agent_eval_cases(str(dataset))


def test_load_agent_eval_cases_rejects_empty_dataset(
    tmp_path: Path,
) -> None:
    dataset = write_text(
        tmp_path / "eval.jsonl",
        "\n# only comment\n",
    )

    with pytest.raises(AgentEvalDatasetError, match="评测集不能为空"):
        load_agent_eval_cases(str(dataset))


def test_load_agent_eval_cases_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_agent_eval_cases("not_exists.jsonl")
