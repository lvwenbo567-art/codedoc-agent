# CodeDoc application layout

This backend is organized by engineering responsibility rather than by study day.

## Main layers

- `api/`: FastAPI routers and response helpers.
- `schemas/`: HTTP request and response schemas.
- `ingestion/`: repository scanning, code parsing, and chunk construction.
- `services/`: retrieval, indexing, RAG, async ingestion, and orchestration services.
- `vectorstores/`: JSON and Qdrant vector store implementations.
- `tools/`: self-owned tool registry, schemas, executors, and CodeDoc tools.
- `function_calling/`: hand-written function-calling loop.
- `langchain_agent/`: LangChain model, tool, and middleware integration.
- `langgraph_agent/`: LangGraph workflows, checkpointing, HITL, and Agent state.
- `memory/`: structured long-term memory and conversation summary support.
- `context_engineering/`: message and evidence budget control.
- `security/`: prompt-injection and sensitive-data handling.
- `evaluation/`: retrieval and Agent evaluation.
- `jobs/`: async ingestion job records, queueing, and lifecycle state.
- `skills/`: project-level Skill definitions and routing.
- `mcp/`: MCP-facing tools, resources, and prompts.

## Compatibility note

The `agents/` package is a small facade for interview-facing structure. The
runtime implementations remain in `function_calling/`, `langchain_agent/`, and
`langgraph_agent/` to avoid breaking tested imports.
