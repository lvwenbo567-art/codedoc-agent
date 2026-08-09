from mcp.mcp_schema import McpCallToolRequest, McpRuntimeConfig
from mcp.tool_adapter import call_mcp_tool, list_mcp_tools


def test_mcp_lists_read_only_tools_only():
    tools = list_mcp_tools(McpRuntimeConfig())
    names = {
        tool.name
        for tool in tools
    }

    assert "get_project_structure" in names
    assert "get_symbol_definition" in names
    assert "read_file_range" in names
    assert "search_code" in names
    assert "search_documents" in names
    assert "run_project_tests" not in names


def test_mcp_can_call_project_structure_tool():
    result = call_mcp_tool(
        McpCallToolRequest(
            tool_name="get_project_structure",
            project_root="test_project",
            arguments={
                "max_depth": 1,
                "max_entries": 30,
                "include_files": True,
                "include_hidden": False,
            },
        )
    )

    assert result.success is True
    assert result.data is not None
    assert result.data["entry_count"] > 0


def test_mcp_rejects_non_read_only_tool():
    result = call_mcp_tool(
        McpCallToolRequest(
            tool_name="run_project_tests",
            arguments={
                "test_path": "tests/test_project_test_tools.py"
            },
        )
    )

    assert result.success is False
    assert result.error_code == "MCP_TOOL_NOT_ALLOWED"
