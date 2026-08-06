from __future__ import annotations

import json

from evaluation.chunk_experiment import (
    evaluate_chunk_case,
    load_chunk_experiment_cases,
    run_chunk_experiment,
)


def test_evaluate_chunk_case_matches_expected_structure() -> None:
    chunks = [
        {
            "chunk_id": "demo.py::foo::part_0",
            "source_path": "demo.py",
            "chunk_type": "code",
            "code_unit_type": "function",
            "symbol_name": "foo",
            "qualified_name": "foo",
            "length": 20,
            "parser": "python_ast",
        }
    ]
    case = {
        "case_id": "foo",
        "source_path": "demo.py",
        "chunk_type": "code",
        "code_unit_type": "function",
        "symbol_name": "foo",
        "qualified_name": "foo",
    }

    result = evaluate_chunk_case(
        case=case,
        chunks=chunks,
    )

    assert result["matched"] is True
    assert result["matched_chunk_id"] == "demo.py::foo::part_0"
    assert result["matched_parser"] == "python_ast"


def test_run_chunk_experiment_summarizes_preservation_rate() -> None:
    chunks = [
        {
            "chunk_id": "demo.py::foo::part_0",
            "source_path": "demo.py",
            "chunk_type": "code",
            "code_unit_type": "function",
            "symbol_name": "foo",
            "qualified_name": "foo",
        }
    ]
    cases = [
        {
            "case_id": "foo",
            "source_path": "demo.py",
            "chunk_type": "code",
            "code_unit_type": "function",
            "symbol_name": "foo",
        },
        {
            "case_id": "bar",
            "source_path": "demo.py",
            "chunk_type": "code",
            "code_unit_type": "function",
            "symbol_name": "bar",
        },
    ]

    report = run_chunk_experiment(
        cases=cases,
        chunks=chunks,
    )

    assert report["case_count"] == 2
    assert report["summary"]["structure_preservation_rate"] == 0.5


def test_load_chunk_experiment_cases(tmp_path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "case_id": "foo",
                "source_path": "demo.py",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_chunk_experiment_cases(str(dataset))

    assert cases == [{"case_id": "foo", "source_path": "demo.py"}]
