from skills import build_builtin_skill_registry


def test_skill_registry_lists_builtin_skills():
    registry = build_builtin_skill_registry()

    names = [
        skill.skill_name
        for skill in registry.list_skills()
    ]

    assert names == [
        "code_navigation",
        "project_onboarding",
        "test_diagnosis",
    ]


def test_skill_router_prefers_code_navigation_for_symbol_query():
    registry = build_builtin_skill_registry()

    result = registry.route(
        "keyword_score 函数在哪里定义？请读取附近源码并解释作用。"
    )

    assert result.skill_name == "code_navigation"
    assert "get_symbol_definition" in result.recommended_tools
    assert "read_file_range" in result.recommended_tools


def test_skill_router_prefers_test_diagnosis_for_pytest_query():
    registry = build_builtin_skill_registry()

    result = registry.route(
        "请运行 tests/test_project_test_tools.py 验证 pytest 是否正常。"
    )

    assert result.skill_name == "test_diagnosis"
    assert "run_project_tests" in result.recommended_tools
