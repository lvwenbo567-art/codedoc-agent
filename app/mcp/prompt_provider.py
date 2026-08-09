from __future__ import annotations

from mcp.mcp_schema import McpPrompt


def list_mcp_prompts() -> list[McpPrompt]:
    return [
        McpPrompt(
            name="project_onboarding",
            description="Explain repository structure and startup path with evidence.",
            template=(
                "Inspect the project structure and documentation. "
                "Summarize main modules, entry points, and startup instructions. "
                "Cite concrete files or tool results."
            ),
        ),
        McpPrompt(
            name="code_navigation",
            description="Locate a symbol and explain nearby implementation.",
            template=(
                "Find the requested function, class, or method. "
                "Read nearby source code before explaining behavior."
            ),
        ),
        McpPrompt(
            name="test_diagnosis",
            description="Diagnose pytest failures with bounded tool execution.",
            template=(
                "Run or inspect the requested test only when allowed. "
                "Summarize pass/fail status and point to likely causes."
            ),
        ),
    ]
