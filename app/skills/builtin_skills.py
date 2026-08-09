from __future__ import annotations

from skills.skill_registry import SkillRegistry
from skills.skill_schema import SkillDefinition, SkillStep


def _project_onboarding_skill() -> SkillDefinition:
    return SkillDefinition(
        skill_name="project_onboarding",
        display_name="Project Onboarding",
        description=(
            "Understand a repository's directory layout, entry points, "
            "documents, and important modules before deeper Q&A."
        ),
        intent_keywords=[
            "项目结构",
            "目录",
            "模块",
            "入口",
            "启动",
            "README",
            "使用文档",
            "overview",
            "structure",
            "onboarding",
        ],
        example_queries=[
            "这个项目有哪些主要目录和模块？",
            "README 或使用文档里有没有说明项目怎么启动？",
        ],
        recommended_tools=[
            "get_project_structure",
            "search_documents",
            "search_code",
        ],
        output_sections=[
            "repository overview",
            "main modules",
            "startup or usage evidence",
            "next suggested questions",
        ],
        steps=[
            SkillStep(
                step_name="inspect_structure",
                description="Read the repository tree and identify major modules.",
                recommended_tools=["get_project_structure"],
            ),
            SkillStep(
                step_name="read_docs",
                description="Search README and usage documents for startup evidence.",
                recommended_tools=["search_documents"],
                required=False,
            ),
        ],
    )


def _code_navigation_skill() -> SkillDefinition:
    return SkillDefinition(
        skill_name="code_navigation",
        display_name="Code Navigation",
        description=(
            "Locate functions, classes, methods, or files, then read nearby "
            "source code to explain implementation details."
        ),
        intent_keywords=[
            "函数",
            "类",
            "方法",
            "定义",
            "源码",
            "实现",
            "在哪里",
            "调用",
            "symbol",
            "definition",
            "source",
        ],
        example_queries=[
            "keyword_score 函数在哪里定义？请读取附近源码并解释作用。",
            "RerankClient score 方法在哪里实现？",
        ],
        recommended_tools=[
            "get_symbol_definition",
            "read_file_range",
            "search_code",
        ],
        output_sections=[
            "definition location",
            "source evidence",
            "implementation explanation",
            "call-chain notes",
        ],
        steps=[
            SkillStep(
                step_name="locate_symbol",
                description="Use exact symbol lookup when a symbol name is present.",
                recommended_tools=["get_symbol_definition"],
            ),
            SkillStep(
                step_name="read_source",
                description="Read the source range around the returned line numbers.",
                recommended_tools=["read_file_range"],
            ),
        ],
    )


def _test_diagnosis_skill() -> SkillDefinition:
    return SkillDefinition(
        skill_name="test_diagnosis",
        display_name="Test Diagnosis",
        description=(
            "Run a bounded pytest command, inspect failures, and explain which "
            "project area needs attention."
        ),
        intent_keywords=[
            "测试",
            "pytest",
            "失败",
            "报错",
            "运行",
            "验证",
            "test",
            "failure",
            "diagnose",
        ],
        example_queries=[
            "请运行 tests/test_project_test_tools.py，验证测试工具是否正常。",
            "这个失败的测试应该怎么定位？",
        ],
        recommended_tools=[
            "run_project_tests",
            "read_file_range",
            "search_code",
        ],
        output_sections=[
            "test command",
            "pass/fail summary",
            "failure diagnosis",
            "suggested fix",
        ],
        requires_human_review_tools=["run_project_tests"],
        steps=[
            SkillStep(
                step_name="request_test_run",
                description="Run a bounded pytest command after approval if configured.",
                recommended_tools=["run_project_tests"],
            ),
            SkillStep(
                step_name="inspect_failure",
                description="Read relevant source or tests when failures appear.",
                recommended_tools=["read_file_range", "search_code"],
                required=False,
            ),
        ],
    )


def build_builtin_skill_registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register(_project_onboarding_skill())
    registry.register(_code_navigation_skill())
    registry.register(_test_diagnosis_skill())
    return registry
