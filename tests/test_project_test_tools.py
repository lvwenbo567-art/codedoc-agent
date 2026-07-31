from __future__ import annotations

from pathlib import Path
import subprocess
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from tools.code_doc_tools import build_code_doc_tool_registry
from tools.executor import ToolExecutor
from tools.models import RunProjectTestsArgs
import tools.project_test_tools as project_test_tools


def test_run_project_tests_schema_descriptions_are_strings() -> None:
    schema = RunProjectTestsArgs.model_json_schema()
    properties = schema["properties"]

    assert isinstance(properties["test_path"]["description"], str)
    assert isinstance(properties["keyword"]["description"], str)
    assert isinstance(properties["max_seconds"]["description"], str)


def test_run_project_tests_builds_safe_pytest_command(
    tmp_path,
    monkeypatch,
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_demo.py").write_text(
        "def test_demo():\n    assert True\n",
        encoding="utf-8",
    )

    captured: dict = {}

    def fake_run(
        command,
        **kwargs,
    ):
        captured["command"] = command
        captured["kwargs"] = kwargs

        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="1 passed\n",
            stderr="",
        )

    monkeypatch.setattr(
        project_test_tools.subprocess,
        "run",
        fake_run,
    )

    registry = build_code_doc_tool_registry(
        project_root=str(tmp_path),
    )
    executor = ToolExecutor(registry)

    result = executor.execute(
        tool_name="run_project_tests",
        arguments={
            "test_path": "tests/test_demo.py",
            "keyword": "demo",
            "max_seconds": 10,
        },
    )

    assert result.success is True
    assert result.data["passed"] is True
    assert result.data["exit_code"] == 0
    assert result.data["timed_out"] is False
    assert result.data["stdout_tail"] == "1 passed\n"
    assert captured["command"][:3] == [
        sys.executable,
        "-m",
        "pytest",
    ]
    assert "--basetemp" in captured["command"]
    basetemp_index = captured["command"].index("--basetemp")
    assert "outputs" in captured["command"][basetemp_index + 1]
    assert "tool_pytest_tmp" in captured["command"][basetemp_index + 1]
    assert captured["command"][-2:] == [
        "-k",
        "demo",
    ]
    assert captured["kwargs"]["cwd"] == str(tmp_path.resolve())
    assert captured["kwargs"]["timeout"] == 10
    assert captured["kwargs"]["check"] is False


def test_run_project_tests_rejects_absolute_path(
    tmp_path,
) -> None:
    registry = build_code_doc_tool_registry(
        project_root=str(tmp_path),
    )
    executor = ToolExecutor(registry)

    result = executor.execute(
        tool_name="run_project_tests",
        arguments={
            "test_path": str(tmp_path / "tests"),
            "max_seconds": 10,
        },
    )

    assert result.success is False
    assert result.error_code == "ABSOLUTE_PATH_FORBIDDEN"


def test_run_project_tests_rejects_path_outside_project(
    tmp_path,
) -> None:
    outside = tmp_path.parent / "outside_tests"
    outside.mkdir(exist_ok=True)

    registry = build_code_doc_tool_registry(
        project_root=str(tmp_path),
    )
    executor = ToolExecutor(registry)

    result = executor.execute(
        tool_name="run_project_tests",
        arguments={
            "test_path": "../outside_tests",
            "max_seconds": 10,
        },
    )

    assert result.success is False
    assert result.error_code == "PATH_OUTSIDE_PROJECT"


def test_run_project_tests_returns_timeout_result(
    tmp_path,
    monkeypatch,
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    def fake_run(
        command,
        **kwargs,
    ):
        raise subprocess.TimeoutExpired(
            cmd=command,
            timeout=kwargs["timeout"],
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.setattr(
        project_test_tools.subprocess,
        "run",
        fake_run,
    )

    registry = build_code_doc_tool_registry(
        project_root=str(tmp_path),
    )
    executor = ToolExecutor(registry)

    result = executor.execute(
        tool_name="run_project_tests",
        arguments={
            "test_path": "tests",
            "max_seconds": 5,
        },
    )

    assert result.success is True
    assert result.data["passed"] is False
    assert result.data["timed_out"] is True
    assert result.data["exit_code"] is None
    assert "partial stdout" in result.data["stdout_tail"]
    assert "partial stderr" in result.data["stderr_tail"]
