from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from security.security_models import PathAccessDeniedError


@dataclass(frozen=True)
class SafePathConfig:
    allowed_suffixes: frozenset[str] = field(
        default_factory=lambda: frozenset({".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"})
    )
    blocked_names: frozenset[str] = field(
        default_factory=lambda: frozenset({".env", ".env.local", "id_rsa", "id_ed25519"})
    )
    blocked_directories: frozenset[str] = field(
        default_factory=lambda: frozenset({".git", ".ssh", ".aws", ".venv", "venv", "node_modules"})
    )
    max_file_bytes: int = 2 * 1024 * 1024
    max_read_lines: int = 1000


class SafeProjectPathResolver:
    """基于 Path.resolve 的项目文件边界校验，防止 ../ 和符号链接逃逸。"""

    def __init__(self, *, project_root: str | Path, config: SafePathConfig | None = None) -> None:
        self.project_root = Path(project_root).resolve(strict=True)
        if not self.project_root.is_dir():
            raise ValueError("project_root 必须是目录")
        self.config = config or SafePathConfig()

    def resolve_file(self, requested_path: str) -> Path:
        if not requested_path or not requested_path.strip():
            raise PathAccessDeniedError("文件路径不能为空")
        requested = Path(requested_path.replace("\\", "/"))
        try:
            candidate = (
                requested.resolve(strict=True)
                if requested.is_absolute()
                else (self.project_root / requested).resolve(strict=True)
            )
        except FileNotFoundError as exc:
            raise PathAccessDeniedError("目标文件不存在或不允许访问") from exc
        try:
            candidate.relative_to(self.project_root)
        except ValueError as exc:
            raise PathAccessDeniedError("路径超出项目根目录范围") from exc
        if not candidate.is_file():
            raise PathAccessDeniedError("目标路径不是文件")
        relative_parts = candidate.relative_to(self.project_root).parts
        if any(part in self.config.blocked_directories for part in relative_parts):
            raise PathAccessDeniedError("禁止访问敏感目录")
        if candidate.name.lower() in self.config.blocked_names:
            raise PathAccessDeniedError("禁止访问敏感文件")
        if candidate.suffix.lower() not in self.config.allowed_suffixes:
            raise PathAccessDeniedError("不允许读取该文件类型")
        if candidate.stat().st_size > self.config.max_file_bytes:
            raise PathAccessDeniedError("文件大小超过安全限制")
        return candidate

    def validate_read_range(self, *, start_line: int, end_line: int) -> None:
        if start_line <= 0 or end_line < start_line:
            raise ValueError("读取行号范围不合法")
        if end_line - start_line + 1 > self.config.max_read_lines:
            raise PathAccessDeniedError("单次读取行数超过安全限制")
