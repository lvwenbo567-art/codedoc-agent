from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_chunk_experiment_cases(path: str) -> list[dict[str, Any]]:
    """
    加载 Chunk 质量评测集。

    每一条 case 描述一个期望被正确保留下来的结构单元，
    例如某个函数、类、方法或文档文件。
    """
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Chunk 评测集不存在：{dataset_path}")

    cases: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()

    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Chunk 评测集第 {line_number} 行不是合法 JSON"
                ) from exc

            case_id = str(case.get("case_id") or "").strip()
            source_path = str(case.get("source_path") or "").strip()

            if not case_id:
                raise ValueError(f"Chunk 评测集第 {line_number} 行缺少 case_id")

            if case_id in seen_case_ids:
                raise ValueError(f"Chunk 评测集存在重复 case_id：{case_id}")

            if not source_path:
                raise ValueError(f"Chunk 评测集第 {line_number} 行缺少 source_path")

            seen_case_ids.add(case_id)
            cases.append(case)

    if not cases:
        raise ValueError("Chunk 评测集不能为空")

    return cases


def load_chunks(path: str) -> list[dict[str, Any]]:
    """
    从 JSON 文件读取已经构建好的 chunks。
    """
    chunks_path = Path(path)

    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks 文件不存在：{chunks_path}")

    data = json.loads(chunks_path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("chunks 文件必须是列表结构")

    return [dict(item) for item in data]


def evaluate_chunk_case(
    *,
    case: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    判断某个结构单元是否被 Chunk 阶段正确保留。
    """
    candidates = [
        chunk
        for chunk in chunks
        if chunk.get("source_path") == case["source_path"]
    ]

    matched = [
        chunk
        for chunk in candidates
        if _matches_expected_fields(
            chunk=chunk,
            case=case,
        )
    ]

    best_chunk = matched[0] if matched else None

    return {
        "case_id": case["case_id"],
        "name": case.get("name", case["case_id"]),
        "source_path": case["source_path"],
        "expected": {
            key: case.get(key)
            for key in (
                "chunk_type",
                "code_unit_type",
                "symbol_name",
                "qualified_name",
            )
            if case.get(key) is not None
        },
        "candidate_count": len(candidates),
        "matched_chunk_id": best_chunk.get("chunk_id") if best_chunk else None,
        "matched": bool(best_chunk),
        "matched_length": best_chunk.get("length") if best_chunk else None,
        "matched_parser": best_chunk.get("parser") if best_chunk else None,
    }


def run_chunk_experiment(
    *,
    cases: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    执行 Chunk 质量实验。
    """
    case_reports = [
        evaluate_chunk_case(
            case=case,
            chunks=chunks,
        )
        for case in cases
    ]

    return {
        "case_count": len(case_reports),
        "chunk_count": len(chunks),
        "summary": summarize_chunk_reports(case_reports),
        "cases": case_reports,
    }


def summarize_chunk_reports(
    case_reports: list[dict[str, Any]],
) -> dict[str, float]:
    """
    汇总 Chunk 结构保留情况。
    """
    if not case_reports:
        return {
            "structure_preservation_rate": 0.0,
            "average_candidate_count": 0.0,
        }

    return {
        "structure_preservation_rate": sum(
            1 for item in case_reports if item["matched"]
        )
        / len(case_reports),
        "average_candidate_count": sum(
            float(item["candidate_count"]) for item in case_reports
        )
        / len(case_reports),
    }


def save_chunk_experiment_report(
    *,
    report: dict[str, Any],
    output_path: str,
) -> str:
    """
    保存 Chunk 实验报告。
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return str(path)


def _matches_expected_fields(
    *,
    chunk: dict[str, Any],
    case: dict[str, Any],
) -> bool:
    for field_name in (
        "chunk_type",
        "code_unit_type",
        "symbol_name",
        "qualified_name",
    ):
        expected_value = case.get(field_name)

        if expected_value is None:
            continue

        if chunk.get(field_name) != expected_value:
            return False

    return True
