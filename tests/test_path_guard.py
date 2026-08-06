from pathlib import Path

import pytest

from security.path_guard import SafeProjectPathResolver
from security.security_models import PathAccessDeniedError


def test_path_guard_allows_project_file_and_blocks_escape(tmp_path: Path) -> None:
    (tmp_path / "src.py").write_text("print('ok')", encoding="utf-8")
    (tmp_path / ".env").write_text("API_KEY=x", encoding="utf-8")
    resolver = SafeProjectPathResolver(project_root=tmp_path)
    assert resolver.resolve_file("src.py") == (tmp_path / "src.py").resolve()
    with pytest.raises(PathAccessDeniedError):
        resolver.resolve_file("../outside.py")
    with pytest.raises(PathAccessDeniedError):
        resolver.resolve_file(".env")
    with pytest.raises(PathAccessDeniedError):
        resolver.validate_read_range(start_line=1, end_line=1001)
