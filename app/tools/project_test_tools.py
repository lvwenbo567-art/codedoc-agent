from __future__ import annotations

import subprocess#它用来启动外部进程
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from tools.errors import ToolBusinessError
from tools.models import RunProjectTestsArgs
from tools.registry import ToolRegistry, ToolSpec


OUTPUT_TAIL_CHARS = 6000


def _normalize_path(value: str) -> str:
    return value.replace("\\", "/").strip()


def _resolve_safe_test_path(
    *,
    project_root: Path,
    test_path: str,
) -> Path:
    normalized = _normalize_path(test_path)
    relative_path = Path(normalized)

    if relative_path.is_absolute():
        raise ToolBusinessError(
            error_code="ABSOLUTE_PATH_FORBIDDEN",
            message="test_path 必须是相对于项目根目录的路径",
        )

    root = project_root.resolve()
    candidate = (root / relative_path).resolve()

    if candidate != root and root not in candidate.parents:
        raise ToolBusinessError(
            error_code="PATH_OUTSIDE_PROJECT",
            message="禁止运行项目根目录之外的测试路径",
        )

    if not candidate.exists():
        raise ToolBusinessError(
            error_code="TEST_PATH_NOT_FOUND",
            message=f"测试路径不存在：{normalized}",
        )

    return candidate


def _tail_text(value: str, max_chars: int = OUTPUT_TAIL_CHARS) -> str:
    if len(value) <= max_chars:
        return value

    return value[-max_chars:]


def _normalize_keyword(keyword: str | None) -> str | None:
    normalized = str(keyword or "").strip()

    if not normalized:
        return None

    if normalized.lower() in {
        "none",
        "null",
        "undefined",
        "nil",
    }:
        return None

    return normalized


def _build_pytest_command(
    *,
    test_path: Path,
    basetemp_path: Path,
    keyword: str | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(test_path),
        "--basetemp",
        str(basetemp_path),
    ]

    normalized_keyword = _normalize_keyword(keyword)

    if normalized_keyword:
        command.extend(
            [
                "-k",
                normalized_keyword,
            ]
        )

    return command


def _run_project_tests(
    *,
    project_root: str,
    test_path: str,
    keyword: str | None,
    max_seconds: int,
) -> dict[str, Any]:
    root = Path(project_root).resolve()

    if not root.exists():
        raise ToolBusinessError(
            error_code="PROJECT_ROOT_NOT_FOUND",
            message=f"项目目录不存在：{root}",
        )

    if not root.is_dir():
        raise ToolBusinessError(
            error_code="PROJECT_ROOT_NOT_DIRECTORY",
            message=f"项目根路径不是目录：{root}",
        )

    resolved_test_path = _resolve_safe_test_path(
        project_root=root,
        test_path=test_path,
    )
    basetemp_path = (
        root
        / "outputs"
        / "tool_pytest_tmp"
        / f"run_{uuid.uuid4().hex}"
    )
    basetemp_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    command = _build_pytest_command(
        test_path=resolved_test_path,
        basetemp_path=basetemp_path,
        keyword=keyword,
    )
    started_at = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration_ms = round(
            (time.perf_counter() - started_at) * 1000,
            2,
        )
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""

        return {
            "passed": False,
            "timed_out": True,
            "exit_code": None,
            "command": " ".join(command),
            "project_root": str(root),
            "test_path": _normalize_path(test_path),
            "basetemp_path": str(basetemp_path),
            "keyword": _normalize_keyword(keyword),
            "max_seconds": max_seconds,
            "duration_ms": duration_ms,
            "stdout_tail": _tail_text(str(stdout)),
            "stderr_tail": _tail_text(str(stderr)),
        }

    duration_ms = round(
        (time.perf_counter() - started_at) * 1000,
        2,
    )

    return {
        "passed": completed.returncode == 0,
        "timed_out": False,
        "exit_code": completed.returncode,
        "command": " ".join(command),
        "project_root": str(root),
        "test_path": _normalize_path(test_path),
        "basetemp_path": str(basetemp_path),
        "keyword": _normalize_keyword(keyword),
        "max_seconds": max_seconds,
        "duration_ms": duration_ms,
        "stdout_tail": _tail_text(completed.stdout or ""),
        "stderr_tail": _tail_text(completed.stderr or ""),
    }


def register_project_test_tools(
    *,
    registry: ToolRegistry,
    project_root: str,
) -> None:
    def run_project_tests(
        test_path: str = "tests",
        keyword: str | None = None,
        max_seconds: int = 60,
    ) -> dict[str, Any]:
        return _run_project_tests(
            project_root=project_root,
            test_path=test_path,
            keyword=keyword,
            max_seconds=max_seconds,
        )

    registry.register(
        ToolSpec(
            name="run_project_tests",
            description=(
                "在受控范围内运行项目 pytest 测试，并返回退出码、耗时和输出摘要。"
                "该工具会启动外部测试进程，适合放入 Human-in-the-loop 审批。"
                "不要把任意 shell 命令传给它，只能通过 test_path、keyword 和 max_seconds 控制 pytest。"
            ),
            args_model=RunProjectTestsArgs,
            handler=run_project_tests,
        )
    )
