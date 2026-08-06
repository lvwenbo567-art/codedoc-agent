from security.tool_security_policy import ToolSecurityPolicy


def test_tool_policy_rejects_oversized_file_read() -> None:
    result = ToolSecurityPolicy().validate_call(
        tool_name="read_file_range",
        arguments={"start_line": 1, "end_line": 2000},
    )
    assert result.allowed is False


def test_tool_policy_allows_bounded_search() -> None:
    assert ToolSecurityPolicy().validate_call(
        tool_name="search_code",
        arguments={"query": "keyword_score", "top_k": 5, "candidate_top_k": 20},
    ).allowed is True
