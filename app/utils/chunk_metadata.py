from __future__ import annotations

from pathlib import Path


def get_file_metadata(project_file: dict) -> tuple[str, str, str, str]:
    """
    从项目文件字典中提取 chunk 构建所需的统一文件元数据。
    """
    source_path = (
        project_file.get("path")
        or project_file.get("source_path")
        or ""
    )

    if not source_path:
        raise ValueError("project_file 缺少 path")

    path = Path(source_path)
    source_name = (
        project_file.get("name")
        or project_file.get("source_name")
        or path.name
    )
    source_suffix = (
        project_file.get("suffix")
        or project_file.get("source_suffix")
        or path.suffix
    )
    content = project_file.get("content", "")

    if not isinstance(content, str):
        raise TypeError(f"文件内容必须是字符串：{source_path}")

    return source_path, source_name, source_suffix, content
