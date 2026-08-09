from __future__ import annotations

from fastapi import APIRouter

from api.api_response import success_response
from mcp.mcp_schema import McpCallToolRequest, McpRuntimeConfig
from mcp.prompt_provider import list_mcp_prompts
from mcp.resource_provider import list_mcp_resources
from mcp.tool_adapter import call_mcp_tool, list_mcp_tools


router = APIRouter(
    prefix="/mcp",
    tags=["mcp"],
)


@router.get("/tools")
def get_mcp_tools() -> dict:
    tools = list_mcp_tools(McpRuntimeConfig())
    return success_response(
        data={
            "tools": [
                tool.model_dump()
                for tool in tools
            ]
        }
    )


@router.post("/tools/call")
def post_mcp_call_tool(request: McpCallToolRequest) -> dict:
    result = call_mcp_tool(request)
    return success_response(data=result.model_dump())


@router.get("/resources")
def get_mcp_resources(project_id: int = 1) -> dict:
    resources = list_mcp_resources(project_id=project_id)
    return success_response(
        data={
            "resources": [
                resource.model_dump()
                for resource in resources
            ]
        }
    )


@router.get("/prompts")
def get_mcp_prompts() -> dict:
    prompts = list_mcp_prompts()
    return success_response(
        data={
            "prompts": [
                prompt.model_dump()
                for prompt in prompts
            ]
        }
    )
