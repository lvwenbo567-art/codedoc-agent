from __future__ import annotations

from pathlib import Path

from config import (
    DEFAULT_CHAT_API_KEY,
    DEFAULT_CHAT_BASE_URL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_PROVIDER,
    DEFAULT_CHAT_TIMEOUT_SECONDS,
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_DEVICE,
    DEFAULT_RERANK_LOCAL_FILES_ONLY,
    DEFAULT_RERANK_MAX_LENGTH,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_PROVIDER,
)
from pipelines.retrieval_pipeline import (
    retrieve_with_rerank,
)
from tools.errors import ToolBusinessError
from tools.code_navigation_tools import register_code_navigation_tools
from tools.models import (
    GetProjectStructureArgs,
    SearchCodeArgs,
    SearchDocumentsArgs,
)
from tools.project_test_tools import register_project_test_tools
from tools.registry import (
    ToolRegistry,
    ToolSpec,
)
from security.path_guard import SafePathConfig


DEFAULT_IGNORED_DIRECTORIES = {
    ".git",
    ".idea",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
}


def _extract_score(
    result: dict,
) -> float:
    score_names = [
        "rerank_score",
        "final_score",
        "multi_query_score",
        "score",
        "vector_score",
    ]

    for score_name in score_names:
        value = result.get(score_name)

        if value is None:
            continue

        try:
            return round(
                float(value),
                6,
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

    return 0.0


def _compact_retrieval_result(
    result: dict,
    max_content_chars: int = 1500,
) -> dict:
    """
    控制 Tool 输出体积。

    不把完整内部检索对象全部发给模型，
    只保留回答所需证据。
    """
    content = str(
        result.get("content", "")
    )

    if len(content) > max_content_chars:
        content = (
            content[:max_content_chars]
            + "\n...[content truncated]"
        )

    return {
        "rank": result.get("rank"),
        "chunk_id": result.get("chunk_id"),
        "source_path": result.get(
            "source_path"
        ),
        "source_name": result.get(
            "source_name"
        ),
        "chunk_type": result.get(
            "chunk_type"
        ),
        "code_unit_type": result.get(
            "code_unit_type"
        ),
        "symbol_name": result.get(
            "symbol_name"
        ),
        "qualified_name": result.get(
            "qualified_name"
        ),
        "parent_class": result.get(
            "parent_class"
        ),
        "signature": result.get(
            "signature"
        ),
        "start_line": result.get(
            "start_line"
        ),
        "end_line": result.get(
            "end_line"
        ),
        "score": _extract_score(result),
        "content": content,
    }


def _run_retrieval_tool(
    *,
    query: str,
    top_k: int,
    candidate_top_k: int,
    query_strategy: str,
    chunk_type: str,
    chunks_path: str,
    index_path: str,
    embedding_provider: str,
    embedding_model: str,
    embedding_base_url: str,
    embedding_api_key: str,
    embedding_timeout_seconds: float,
    mock_dimension: int,
    rerank_provider: str,
    rerank_model: str,
    rerank_device: str,
    rerank_batch_size: int,
    rerank_max_length: int,
    rerank_local_files_only: bool,
    query_rewrite_provider: str,
    query_rewrite_model: str,
    query_rewrite_base_url: str,
    query_rewrite_api_key: str,
    query_rewrite_timeout_seconds: float,
) -> dict:
    chunks_file = Path(chunks_path)
    index_file = Path(index_path)

    if not chunks_file.exists():
        raise ToolBusinessError(
            error_code="CHUNKS_NOT_FOUND",
            message=(
                "Chunk 文件不存在："
                f"{chunks_file}"
            ),
        )

    if not index_file.exists():
        raise ToolBusinessError(
            error_code="INDEX_NOT_FOUND",
            message=(
                "向量索引不存在："
                f"{index_file}"
            ),
        )

    retrieval_result = (
        retrieve_with_rerank(
            query=query,
            chunks_path=str(chunks_file),
            index_path=str(index_file),
            candidate_top_k=(
                candidate_top_k
            ),
            final_top_k=top_k,
            chunk_type=chunk_type,
            query_strategy=query_strategy,
            rewrite_count=2,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            embedding_timeout_seconds=embedding_timeout_seconds,
            mock_dimension=mock_dimension,
            rerank_provider=rerank_provider,
            rerank_model=rerank_model,
            rerank_device=rerank_device,
            rerank_batch_size=rerank_batch_size,
            rerank_max_length=rerank_max_length,
            rerank_local_files_only=rerank_local_files_only,
            query_rewrite_provider=query_rewrite_provider,
            query_rewrite_model=query_rewrite_model,
            query_rewrite_base_url=query_rewrite_base_url,
            query_rewrite_api_key=query_rewrite_api_key,
            query_rewrite_timeout_seconds=query_rewrite_timeout_seconds,
        )
    )

    raw_results = retrieval_result.get(
        "results",
        [],
    )

    if not isinstance(raw_results, list):
        raise ToolBusinessError(
            error_code=(
                "INVALID_RETRIEVAL_RESULT"
            ),
            message=(
                "检索服务返回结果中缺少 results 列表"
            ),
        )

    compact_results = [
        _compact_retrieval_result(result)
        for result in raw_results
    ]

    return {
        "query": query,
        "chunk_type": chunk_type,
        "query_strategy": query_strategy,
        "retrieval_mode": retrieval_result.get(
            "retrieval_mode"
        ),
        "degraded": retrieval_result.get(
            "degraded",
            False,
        ),
        "degrade_reason": retrieval_result.get(
            "degrade_reason"
        ),
        "candidate_count": retrieval_result.get(
            "candidate_count"
        ),
        "result_count": len(
            compact_results
        ),
        "results": compact_results,
    }


def _build_project_structure(
    *,
    project_root: str,
    max_depth: int,
    max_entries: int,
    include_files: bool,
    include_hidden: bool,
) -> dict:
    root = Path(project_root).resolve()

    if not root.exists():
        raise ToolBusinessError(
            error_code="PROJECT_ROOT_NOT_FOUND",
            message=(
                "项目目录不存在："
                f"{root}"
            ),
        )

    if not root.is_dir():
        raise ToolBusinessError(
            error_code="PROJECT_ROOT_NOT_DIRECTORY",
            message=(
                "项目根路径不是目录："
                f"{root}"
            ),
        )

    entries: list[dict] = []
    path_config = SafePathConfig()

    def should_skip(path: Path) -> bool:
        if (
            not include_hidden
            and path.name.startswith(".")
        ):
            return True

        if path.name.lower() in path_config.blocked_names:
            return True

        if path.is_file() and path.suffix.lower() not in path_config.allowed_suffixes:
            return True

        return (
            path.is_dir()
            and path.name
            in (DEFAULT_IGNORED_DIRECTORIES | path_config.blocked_directories)
        )

    # 采用广度优先遍历：目录问答通常先需要顶层模块，而不是某个目录下的
    # 大量子文件。原先的深度优先遍历会让 app/ 等大目录耗尽 max_entries，
    # 导致 docs、tests 等同级目录没有机会出现在结果中。
    pending_directories: list[tuple[Path, int]] = [(root, 1)]

    while pending_directories and len(entries) < max_entries:
        directory, depth = pending_directories.pop(0)
        if depth > max_depth:
            continue

        try:
            children = sorted(
                directory.iterdir(),
                key=lambda item: (
                    not item.is_dir(),
                    item.name.lower(),
                ),
            )
        except OSError:
            # Windows 上可能遗留被 pytest 或其他进程占用的目录；目录展示是
            # 尽力而为的只读能力，跳过不可访问节点不应中断整个 Agent。
            continue

        for child in children:
            if len(entries) >= max_entries:
                break

            try:
                if should_skip(child):
                    continue
            except OSError:
                continue

            if child.is_file() and not include_files:
                continue

            relative_path = child.relative_to(root).as_posix()
            entries.append(
                {
                    "path": relative_path,
                    "name": child.name,
                    "type": "directory" if child.is_dir() else "file",
                    "depth": depth,
                }
            )

            if child.is_dir() and depth < max_depth:
                pending_directories.append((child, depth + 1))

    return {
        "project_root": str(root),
        "max_depth": max_depth,
        "max_entries": max_entries,
        "include_files": include_files,
        "include_hidden": include_hidden,
        "entry_count": len(entries),
        "truncated": len(entries) >= max_entries,
        "entries": entries,
    }


def build_code_doc_tool_registry(
    *,
    project_root: str = ".",
    chunks_path: str = "outputs/chunks.json",
    index_path: str = "outputs/vector_index.json",
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL,
    embedding_api_key: str = DEFAULT_EMBEDDING_API_KEY,
    embedding_timeout_seconds: float = DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    mock_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    rerank_provider: str = DEFAULT_RERANK_PROVIDER,
    rerank_model: str = DEFAULT_RERANK_MODEL,
    rerank_device: str = DEFAULT_RERANK_DEVICE,
    rerank_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    rerank_max_length: int = DEFAULT_RERANK_MAX_LENGTH,
    rerank_local_files_only: bool = DEFAULT_RERANK_LOCAL_FILES_ONLY,
    query_rewrite_provider: str = DEFAULT_CHAT_PROVIDER,
    query_rewrite_model: str = DEFAULT_CHAT_MODEL,
    query_rewrite_base_url: str = DEFAULT_CHAT_BASE_URL,
    query_rewrite_api_key: str = DEFAULT_CHAT_API_KEY,
    query_rewrite_timeout_seconds: float = DEFAULT_CHAT_TIMEOUT_SECONDS,
) -> ToolRegistry:
    """
    构建 CodeDoc 项目的工具注册表。

    project_root、chunks_path 和 index_path 由应用端绑定，
    不允许模型通过工具参数任意修改。
    """
    registry = ToolRegistry()

    def search_code(
        query: str,
        top_k: int = 5,
        candidate_top_k: int = 20,
        query_strategy: str = "multi_query",
    ) -> dict:
        return _run_retrieval_tool(
            query=query,
            top_k=top_k,
            candidate_top_k=candidate_top_k,
            query_strategy=query_strategy,
            chunk_type="code",
            chunks_path=chunks_path,
            index_path=index_path,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            embedding_timeout_seconds=embedding_timeout_seconds,
            mock_dimension=mock_dimension,
            rerank_provider=rerank_provider,
            rerank_model=rerank_model,
            rerank_device=rerank_device,
            rerank_batch_size=rerank_batch_size,
            rerank_max_length=rerank_max_length,
            rerank_local_files_only=rerank_local_files_only,
            query_rewrite_provider=query_rewrite_provider,
            query_rewrite_model=query_rewrite_model,
            query_rewrite_base_url=query_rewrite_base_url,
            query_rewrite_api_key=query_rewrite_api_key,
            query_rewrite_timeout_seconds=query_rewrite_timeout_seconds,
        )

    def search_documents(
        query: str,
        top_k: int = 5,
        candidate_top_k: int = 20,
        query_strategy: str = "multi_query",
    ) -> dict:
        return _run_retrieval_tool(
            query=query,
            top_k=top_k,
            candidate_top_k=candidate_top_k,
            query_strategy=query_strategy,
            chunk_type="document",
            chunks_path=chunks_path,
            index_path=index_path,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            embedding_timeout_seconds=embedding_timeout_seconds,
            mock_dimension=mock_dimension,
            rerank_provider=rerank_provider,
            rerank_model=rerank_model,
            rerank_device=rerank_device,
            rerank_batch_size=rerank_batch_size,
            rerank_max_length=rerank_max_length,
            rerank_local_files_only=rerank_local_files_only,
            query_rewrite_provider=query_rewrite_provider,
            query_rewrite_model=query_rewrite_model,
            query_rewrite_base_url=query_rewrite_base_url,
            query_rewrite_api_key=query_rewrite_api_key,
            query_rewrite_timeout_seconds=query_rewrite_timeout_seconds,
        )

    def get_project_structure(
        max_depth: int = 4,
        max_entries: int = 300,
        include_files: bool = True,
        include_hidden: bool = False,
    ) -> dict:
        return _build_project_structure(
            project_root=project_root,
            max_depth=max_depth,
            max_entries=max_entries,
            include_files=include_files,
            include_hidden=include_hidden,
        )

    registry.register(
        ToolSpec(
            name="search_code",
            description=(
                "检索项目中的 Python 代码、函数、类、"
                "方法、调用关系和实现位置。"
            ),
            args_model=SearchCodeArgs,
            handler=search_code,
        )
    )

    registry.register(
        ToolSpec(
            name="search_documents",
            description=(
                "检索项目 README、Markdown 文档、"
                "使用说明和非代码文本内容。"
            ),
            args_model=SearchDocumentsArgs,
            handler=search_documents,
        )
    )

    registry.register(
        ToolSpec(
            name="get_project_structure",
            description=(
                "查看项目目录结构、主要模块和文件分布。"
            ),
            args_model=GetProjectStructureArgs,
            handler=get_project_structure,
        )
    )

    register_code_navigation_tools(
        registry=registry,
        project_root=project_root,
        chunks_path=chunks_path,
    )

    register_project_test_tools(
        registry=registry,
        project_root=project_root,
    )

    return registry
