from __future__ import annotations

from mcp.mcp_schema import McpResource


def list_mcp_resources(project_id: int = 1) -> list[McpResource]:
    return [
        McpResource(
            uri=f"codedoc://project/{project_id}/structure",
            name="Project structure",
            description="Repository tree and module layout for a CodeDoc project.",
        ),
        McpResource(
            uri=f"codedoc://project/{project_id}/chunks",
            name="Indexed chunks",
            description="Code and document chunks produced by ingestion.",
        ),
        McpResource(
            uri=f"codedoc://project/{project_id}/vector-index",
            name="Vector index",
            description="Embedding vectors and chunk metadata used for retrieval.",
        ),
        McpResource(
            uri=f"codedoc://project/{project_id}/evaluation",
            name="Evaluation reports",
            description="Retrieval and Agent evaluation reports for regression analysis.",
        ),
    ]
