from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1] / "app"
    )
)


from langchain_agent.runtime_context import (
    CodeDocRuntimeContext,
    build_effective_thread_id,
)
from tools.code_doc_tools import build_code_doc_tool_registry


def test_runtime_context_and_effective_thread_id():
    context = CodeDocRuntimeContext(
        user_id="user-1",
        project_id=12,
        project_root=".",
        chunks_path="outputs/chunks.json",
        index_path="outputs/vector_index.json",
        run_id="run-1",
        trace_id="trace-1",
    )

    assert context.permissions == ("project:read",)
    assert (
        build_effective_thread_id(
            project_id=12,
            thread_id="chat-001",
        )
        == "project:12:thread:chat-001"
    )


def test_runtime_context_not_in_tool_schema(tmp_path):
    chunks_path = tmp_path / "chunks.json"
    index_path = tmp_path / "index.json"
    chunks_path.write_text("[]", encoding="utf-8")
    index_path.write_text("{}", encoding="utf-8")

    registry = build_code_doc_tool_registry(
        project_root=str(tmp_path),
        chunks_path=str(chunks_path),
        index_path=str(index_path),
    )
    forbidden_fields = {
        "user_id",
        "project_id",
        "project_root",
        "permissions",
        "run_id",
        "trace_id",
    }

    for tool_spec in registry.list_specs():
        schema = tool_spec.args_model.model_json_schema()
        properties = set(
            schema.get("properties", {}).keys()
        )

        assert properties.isdisjoint(forbidden_fields)
