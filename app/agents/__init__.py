"""
Facade package for CodeDoc Agent implementations.

The physical implementations remain in:
- function_calling
- langchain_agent
- langgraph_agent

This package gives the project a clearer interview-facing structure without
breaking existing imports or tested runtime paths.
"""

__all__: list[str] = []
