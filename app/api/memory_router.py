from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from api.api_response import success_response
from langgraph_agent.thread_identity import build_effective_thread_id, validate_public_thread_id
from memory.conversation_summary_service import ConversationSummaryService
from memory.memory_models import CreateMemoryInput, UpdateMemoryInput
from memory.memory_policy import MemoryWritePolicy
from memory.memory_repository import MemoryNotFoundError, MemoryRepository


router = APIRouter(prefix="/memory", tags=["memory"])


class StrictMemoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ThreadSummaryRequest(StrictMemoryRequest):
    user_id: str = Field(default="local-user", min_length=1, max_length=120)
    project_id: int = Field(ge=1)
    thread_id: str = Field(min_length=1, max_length=120)


class ScopedMemoryRequest(StrictMemoryRequest):
    user_id: str = Field(default="local-user", min_length=1, max_length=120)
    project_id: int = Field(ge=1)


class CreateMemoryRequest(CreateMemoryInput):
    pass


class UpdateMemoryRequest(UpdateMemoryInput):
    user_id: str = Field(default="local-user", min_length=1, max_length=120)
    project_id: int = Field(ge=1)


def get_memory_repository(request: Request) -> MemoryRepository:
    repository = getattr(request.app.state, "memory_repository", None)
    if repository is None:
        raise RuntimeError("Memory Repository 未初始化")
    return repository


@router.get("/threads/{thread_id}/summary")
async def get_thread_summary(thread_id: str, request: Request, user_id: str = Query(default="local-user"), project_id: int = Query(ge=1)) -> dict:
    validate_public_thread_id(thread_id)
    effective_thread_id = build_effective_thread_id(project_id=project_id, thread_id=thread_id)
    record = await get_memory_repository(request).get_summary(effective_thread_id=effective_thread_id)
    return success_response(data=record.model_dump() if record else {"exists": False, "user_id": user_id, "project_id": project_id, "thread_id": thread_id, "effective_thread_id": effective_thread_id})


@router.post("/threads/{thread_id}/summarize")
async def summarize_thread(thread_id: str, body: ThreadSummaryRequest, request: Request) -> dict:
    if body.thread_id != thread_id:
        raise HTTPException(status_code=400, detail="路径 thread_id 与请求体不一致")
    validate_public_thread_id(thread_id)
    effective_thread_id = build_effective_thread_id(project_id=body.project_id, thread_id=thread_id)
    checkpointer = request.app.state.checkpoint_runtime.checkpointer
    checkpoint = await checkpointer.aget_tuple({"configurable": {"thread_id": effective_thread_id}})
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="未找到该线程的 Checkpoint")
    values = checkpoint.checkpoint.get("channel_values", {})
    messages = list(values.get("messages") or []) if isinstance(values, dict) else []
    repository = get_memory_repository(request)
    previous = await repository.get_summary(effective_thread_id=effective_thread_id)
    service = ConversationSummaryService(model=None)
    plan = service.build_update_plan(messages=messages, covered_message_count=previous.covered_message_count if previous else 0)
    if not plan.should_update:
        return success_response(data={"updated": False, "reason": "当前历史未达到摘要阈值或没有新增可摘要完整回合"})
    summary = service.summarize(previous=previous.summary if previous else None, messages=plan.source_messages)
    record = await repository.upsert_summary(user_id=body.user_id, project_id=body.project_id, thread_id=thread_id,
                                             effective_thread_id=effective_thread_id, summary=summary,
                                             covered_turn_count=plan.covered_turn_count,
                                             covered_message_count=plan.covered_message_count,
                                             source_message_count=len(messages))
    return success_response(data={"updated": True, "summary": record.model_dump()})


@router.post("/items")
async def create_memory_item(body: CreateMemoryRequest, request: Request) -> dict:
    policy = MemoryWritePolicy().validate_manual_content(body.content)
    if not policy.allowed:
        raise HTTPException(status_code=400, detail=policy.error_message)
    data = body.model_copy(update={"content": policy.content})
    item = await get_memory_repository(request).create_memory(data)
    return success_response(data=item.model_dump())


@router.get("/items")
async def list_memory_items(request: Request, user_id: str = Query(default="local-user"), project_id: int = Query(ge=1),
                            thread_id: str | None = Query(default=None), query: str | None = Query(default=None),
                            memory_type: str | None = Query(default=None), include_inactive: bool = Query(default=False),
                            limit: int = Query(default=20, ge=1, le=100)) -> dict:
    items = await get_memory_repository(request).list_memories(user_id=user_id, project_id=project_id, thread_id=thread_id,
                                                                 query=query, memory_type=memory_type,
                                                                 include_inactive=include_inactive, limit=limit)
    return success_response(data={"count": len(items), "items": [item.model_dump() for item in items]})


@router.patch("/items/{memory_id}")
async def update_memory_item(memory_id: str, body: UpdateMemoryRequest, request: Request) -> dict:
    if body.content is not None:
        policy = MemoryWritePolicy().validate_manual_content(body.content)
        if not policy.allowed:
            raise HTTPException(status_code=400, detail=policy.error_message)
        body = body.model_copy(update={"content": policy.content})
    try:
        item = await get_memory_repository(request).update_memory(memory_id=memory_id, user_id=body.user_id,
                                                                    project_id=body.project_id, value=body)
    except MemoryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return success_response(data=item.model_dump())


@router.delete("/items/{memory_id}")
async def delete_memory_item(memory_id: str, body: ScopedMemoryRequest, request: Request) -> dict:
    try:
        item = await get_memory_repository(request).delete_memory(memory_id=memory_id, user_id=body.user_id,
                                                                    project_id=body.project_id)
    except MemoryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return success_response(data=item.model_dump())
