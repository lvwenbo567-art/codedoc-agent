from __future__ import annotations

import asyncio#支持异步调用和同步转异步
import inspect#检查 agent.invoke 是否支持 context 参数
import json
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from config import (
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_DEVICE,
    DEFAULT_RERANK_LOCAL_FILES_ONLY,
    DEFAULT_RERANK_MAX_LENGTH,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_PROVIDER,
)
from langchain_agent.agent_factory import create_codedoc_agent
from langchain_agent.agent_runtime import AgentRuntime
from langchain_agent.agent_runtime_cache import (
    GLOBAL_AGENT_RUNTIME_CACHE,
    AgentRuntimeCache,
    build_project_runtime_key,
)
from langchain_agent.agent_schema import LangChainAgentResult, LangChainToolTrace
from langchain_agent.chat_service import extract_message_text
from langchain_agent.checkpointer_factory import create_in_memory_checkpointer
from langchain_agent.conversation_service import resolve_effective_thread_id
from langchain_agent.middleware_config import LangChainMiddlewareConfig
from langchain_agent.middleware_factory import build_agent_middleware
from langchain_agent.model_config import LangChainModelConfig
from langchain_agent.runtime_context import CodeDocRuntimeContext
from langchain_agent.trace_recorder import AgentTraceRecorder, utc_now
from langchain_agent.tool_adapter import build_langchain_tools
from tools.code_doc_tools import build_code_doc_tool_registry
from tools.executor import ToolExecutor


class LangChainAgentExecutionError(RuntimeError):
    pass


def _content_to_text(content: Any) -> str:#把 ToolMessage 或 AIMessage 里的 content 转成字符串。
    if isinstance(content, str):
        return content

    try:
        return json.dumps(
            content,
            ensure_ascii=False,
        )
    except TypeError:
        return str(content)


def _parse_tool_result(raw_output: str) -> dict:#把工具返回的字符串解析成 dict。
    try:
        value = json.loads(raw_output)

        if isinstance(value, dict):
            return value

    except json.JSONDecodeError:
        pass

    if raw_output.startswith("Error invoking tool"):
        return {
            "success": False,
            "error_code": "LANGCHAIN_TOOL_INVOCATION_ERROR",
            "error_message": raw_output,
        }

    return {}


def _extract_tool_traces(messages: list[BaseMessage]) -> list[LangChainToolTrace]:
    #从 Agent messages 里提取工具调用记录。
    pending_calls: dict[str, dict] = {}
    traces: list[LangChainToolTrace] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls or []:
                call_id = str(tool_call.get("id") or "")

                if not call_id:
                    continue

                args = tool_call.get("args", {})

                if not isinstance(args, dict):
                    args = {
                        "raw": args,
                    }

                pending_calls[call_id] = {
                    "tool_name": str(tool_call.get("name") or ""),
                    "arguments": args,
                }

        elif isinstance(message, ToolMessage):
            call_id = str(message.tool_call_id or "")
            raw_output = _content_to_text(message.content)
            parsed_result = _parse_tool_result(raw_output)
            pending = pending_calls.get(call_id, {})

            traces.append(
                LangChainToolTrace(
                    tool_call_id=call_id,
                    tool_name=(
                        str(parsed_result.get("tool_name") or "")
                        or str(getattr(message, "name", "") or "")
                        or str(pending.get("tool_name", ""))
                    ),
                    arguments=pending.get("arguments", {}),
                    success=parsed_result.get("success"),
                    error_code=parsed_result.get("error_code"),
                    error_message=parsed_result.get("error_message"),
                    duration_ms=parsed_result.get("duration_ms"),
                    raw_output=raw_output,
                )
            )

    return traces


def _extract_final_answer(messages: list[BaseMessage]) -> str:
    #从最后一个 AIMessage 里拿最终回答。
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            answer = extract_message_text(message)

            if answer:
                return answer

    raise LangChainAgentExecutionError(
        "Agent 没有返回最终 AIMessage 回答"
    )


def _count_current_turn_messages(messages: list[BaseMessage]) -> tuple[int, int]:
    #def _count_current_turn_messages(messages: list[BaseMessage]) -> tuple[int, int]:
    last_human_index = -1

    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            last_human_index = index

    if last_human_index < 0:
        return len(messages), 0

    history_count = last_human_index
    current_turn_count = len(messages) - last_human_index

    return history_count, current_turn_count


def _supports_parameter(func: Any, parameter_name: str) -> bool:
    #检查 agent.invoke / agent.ainvoke 是否支持 context 参数。
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True

    return parameter_name in signature.parameters


class LangChainAgentService:
    """
    LangChain CodeDoc Agent 服务。

    Day34 在 Day33 Trace 基础上加入 Runtime Context、thread_id 短期记忆、
    InMemorySaver Checkpointer、Runtime Cache 和 async 调用。
    """

    def __init__(
        self,
        *,
        config: LangChainModelConfig | None = None,
        model_config: LangChainModelConfig | None = None,
        middleware_config: LangChainMiddlewareConfig | None = None,
        runtime_cache: AgentRuntimeCache | None = None,
        project_root: str = ".",
        chunks_path: str = "outputs/chunks.json",
        index_path: str = "outputs/vector_index.json",
        recursion_limit: int = 20,
        agent: Any | None = None,
        embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL,
        embedding_api_key: str = DEFAULT_EMBEDDING_API_KEY,
        embedding_timeout_seconds: float = DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
        mock_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        rerank_provider: str = DEFAULT_RERANK_PROVIDER,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        rerank_device: str = DEFAULT_RERANK_DEVICE,
        rerank_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
        rerank_max_length: int = DEFAULT_RERANK_MAX_LENGTH,
        rerank_local_files_only: bool = DEFAULT_RERANK_LOCAL_FILES_ONLY,
    ) -> None:
        resolved_config = config or model_config

        if resolved_config is None:
            raise ValueError("必须提供 LangChainModelConfig")

        self.config = resolved_config
        self.middleware_config = middleware_config or LangChainMiddlewareConfig.from_env()
        self.runtime_cache = runtime_cache or GLOBAL_AGENT_RUNTIME_CACHE
        self.project_root = project_root
        self.chunks_path = chunks_path
        self.index_path = index_path
        self.recursion_limit = recursion_limit
        self._agent = agent
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_base_url = embedding_base_url
        self.embedding_api_key = embedding_api_key
        self.embedding_timeout_seconds = embedding_timeout_seconds
        self.mock_dimension = mock_dimension
        self.rerank_provider = rerank_provider
        self.rerank_model = rerank_model
        self.rerank_device = rerank_device
        self.rerank_batch_size = rerank_batch_size
        self.rerank_max_length = rerank_max_length
        self.rerank_local_files_only = rerank_local_files_only
        self._tool_count = 0

    def _validate_paths(self) -> None:
        project_root = Path(self.project_root)

        if not project_root.exists():
            raise FileNotFoundError(f"项目目录不存在：{project_root}")

        if not project_root.is_dir():
            raise NotADirectoryError(f"project_root 不是目录：{project_root}")

        chunks_path = Path(self.chunks_path)

        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunk 文件不存在：{chunks_path}")

    def _build_tools(self) -> list:
        registry = build_code_doc_tool_registry(
            project_root=self.project_root,
            chunks_path=self.chunks_path,
            index_path=self.index_path,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
            embedding_base_url=self.embedding_base_url,
            embedding_api_key=self.embedding_api_key,
            embedding_timeout_seconds=self.embedding_timeout_seconds,
            mock_dimension=self.mock_dimension,
            rerank_provider=self.rerank_provider,
            rerank_model=self.rerank_model,
            rerank_device=self.rerank_device,
            rerank_batch_size=self.rerank_batch_size,
            rerank_max_length=self.rerank_max_length,
            rerank_local_files_only=self.rerank_local_files_only,
            query_rewrite_provider=self.config.provider,
            query_rewrite_model=self.config.model_name,
            query_rewrite_base_url=self.config.base_url,
            query_rewrite_api_key=self.config.api_key.get_secret_value(),
            query_rewrite_timeout_seconds=self.config.timeout_seconds,
        )
        executor = ToolExecutor(registry)
        tools = build_langchain_tools(
            registry=registry,
            executor=executor,
        )
        self._tool_count = len(tools)

        return tools

    def _build_runtime(
        self,
        *,
        project_id: int,
    ) -> AgentRuntime:#拿到当前请求应该使用的 AgentRuntime。
        if self._agent is not None:
            checkpointer = create_in_memory_checkpointer()
            return AgentRuntime(
                agent=self._agent,
                checkpointer=checkpointer,
                tool_count=self._tool_count,
            )

        tools = self._build_tools()
        key = build_project_runtime_key(
            project_id=project_id,
            project_root=self.project_root,
            chunks_path=self.chunks_path,
            index_path=self.index_path,
            model_config=self.config,
        )

        runtime = self.runtime_cache.get_or_create(
            key=key,
            tools=tools,
            create_agent_func=lambda checkpointer: create_codedoc_agent(
                config=self.config,
                tools=tools,
                checkpointer=checkpointer,
            ),
        )
        self._tool_count = runtime.tool_count

        return runtime

    def _record_tool_traces(
        self,
        *,
        recorder: AgentTraceRecorder,
        tool_traces: list[LangChainToolTrace],
    ) -> None:
        for trace in tool_traces:
            now = utc_now()
            recorder.add_tool_call(
                tool_call_id=trace.tool_call_id,
                tool_name=trace.tool_name,
                arguments=trace.arguments,
                started_at=now,
                completed_at=now,
                duration_ms=trace.duration_ms or 0.0,
                success=bool(trace.success),
                result_preview=trace.raw_output,
            )

    def _build_invoke_config(
        self,
        *,
        effective_thread_id: str,
        recorder: AgentTraceRecorder,
    ) -> dict:
        return {
            "recursion_limit": self.recursion_limit,#限制 Agent 最多执行多少步。
            "configurable": {
                "thread_id": effective_thread_id,
            },
            '''
memory = checkpointer.load("project:1:thread:chat-001")
然后执行完以后，再保存回去：
checkpointer.save("project:1:thread:chat-001", new_state)
当然这两行不是你手写的，是 LangGraph 内部做的。
'''
            "metadata": {
                "run_id": recorder.run_id,
                "trace_id": recorder.trace_id,
            },
        }

    def _build_runtime_context(
        self,
        *,
        user_id: str,
        project_id: int,
        run_id: str,
        trace_id: str,
    ) -> CodeDocRuntimeContext:
        return CodeDocRuntimeContext(
            user_id=user_id,
            project_id=project_id,
            project_root=self.project_root,
            chunks_path=self.chunks_path,
            index_path=self.index_path,
            run_id=run_id,
            trace_id=trace_id,
        )

    def _invoke_agent(
        self,
        *,
        agent: Any,
        query: str,
        invoke_config: dict,
        runtime_context: CodeDocRuntimeContext,
    ) -> dict:
        payload = {
            "messages": [
                HumanMessage(content=query),
            ]
        }

        if _supports_parameter(agent.invoke, "context"):
            return agent.invoke(
                payload,
                config=invoke_config,
                context=runtime_context,
            )

        return agent.invoke(
            payload,
            config=invoke_config,
        )

    async def _ainvoke_agent(
        self,
        *,
        agent: Any,
        query: str,
        invoke_config: dict,
        runtime_context: CodeDocRuntimeContext,
    ) -> dict:
        payload = {
            "messages": [
                HumanMessage(content=query),
            ]
        }

        if hasattr(agent, "ainvoke"):
            if _supports_parameter(agent.ainvoke, "context"):
                return await agent.ainvoke(
                    payload,
                    config=invoke_config,
                    context=runtime_context,
                )

            return await agent.ainvoke(
                payload,
                config=invoke_config,
            )

        return await asyncio.to_thread(#把同步调用放到单独线程里跑。也就是：异步接口里兼容同步 Agent。
            self._invoke_agent,
            agent=agent,
            query=query,
            invoke_config=invoke_config,
            runtime_context=runtime_context,
        )

    def _build_result(
        self,
        *,
        query: str,
        project_id: int,
        thread_id: str,
        effective_thread_id: str,
        recorder: AgentTraceRecorder,
        start_time: float,
        messages: list[BaseMessage],
        answer: str,
        tool_traces: list[LangChainToolTrace],
    ) -> LangChainAgentResult:
        recorder.add_model_call(
            started_at=recorder.snapshot().started_at,
            completed_at=utc_now(),
            duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
            message_count=len(messages),
            available_tool_count=self._tool_count,
            success=True,
        )
        self._record_tool_traces(
            recorder=recorder,
            tool_traces=tool_traces,
        )

        success = True
        stop_reason = "completed"
        trace_status = "completed"

        if len(tool_traces) > self.middleware_config.tool_run_limit:
            success = False
            stop_reason = "tool_call_limit"
            trace_status = "limited"
            answer = "Agent 已达到本次请求的工具调用次数限制。"

        recorder.finish(
            status=trace_status,
            stop_reason=stop_reason,
        )
        trace = recorder.snapshot()
        history_count, current_turn_count = _count_current_turn_messages(messages)

        return LangChainAgentResult(
            run_id=trace.run_id,
            trace_id=trace.trace_id,
            project_id=project_id,
            thread_id=thread_id,
            effective_thread_id=effective_thread_id,
            query=query,
            answer=answer,
            success=success,
            degraded=trace.degraded,
            provider=self.config.provider,
            model_name=self.config.model_name,
            stop_reason=stop_reason,
            message_count=len(messages),
            history_message_count=history_count,
            current_turn_message_count=current_turn_count,
            model_call_count=len(trace.model_calls),
            tool_call_count=len(tool_traces),
            message_trim_count=len(trace.message_trims),
            tool_traces=tool_traces,
            trace=trace,
            total_duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
        )

    def _build_limited_result(
        self,
        *,
        query: str,
        project_id: int,
        thread_id: str,
        effective_thread_id: str,
        recorder: AgentTraceRecorder,
        start_time: float,
        error: Exception,
    ) -> LangChainAgentResult:
        recorder.finish(
            status="limited",
            stop_reason="recursion_limit",
            error=error,
        )
        trace = recorder.snapshot()

        return LangChainAgentResult(
            run_id=trace.run_id,
            trace_id=trace.trace_id,
            project_id=project_id,
            thread_id=thread_id,
            effective_thread_id=effective_thread_id,
            query=query,
            answer="Agent 达到 Graph recursion_limit，可能存在循环调用。",
            success=False,
            degraded=trace.degraded,
            provider=self.config.provider,
            model_name=self.config.model_name,
            stop_reason="recursion_limit",
            message_count=1,
            history_message_count=0,
            current_turn_message_count=1,
            model_call_count=len(trace.model_calls),
            tool_call_count=0,
            message_trim_count=len(trace.message_trims),
            tool_traces=[],
            trace=trace,
            total_duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
        )

    def _prepare_run(
        self,
        *,
        query: str,
        project_id: int,
        thread_id: str,
        user_id: str,
        project_root: str | None,
        chunks_path: str | None,
        index_path: str | None,
        recursion_limit: int | None,
        run_id: str | None,
        trace_id: str | None,
    ) -> tuple[AgentRuntime, AgentTraceRecorder, str, dict, CodeDocRuntimeContext]:
        #把一次请求准备成 Agent 可以执行的所有材料。
        query = query.strip()

        if not query:
            raise ValueError("query 不能为空")

        if project_root is not None:
            self.project_root = project_root

        if chunks_path is not None:
            self.chunks_path = chunks_path

        if index_path is not None:
            self.index_path = index_path

        if recursion_limit is not None:
            self.recursion_limit = recursion_limit

        self._validate_paths()
        recorder = AgentTraceRecorder(
            run_id=run_id,
            trace_id=trace_id,
        )
        middleware_bundle = build_agent_middleware(
            config=self.middleware_config,
            recorder=recorder,
            primary_model_config=self.config,
        )
        middleware_bundle.fallback.mark_if_enabled()

        runtime = self._build_runtime(
            project_id=project_id,
        )
        effective_thread_id = resolve_effective_thread_id(
            project_id=project_id,
            thread_id=thread_id,
        )
        invoke_config = self._build_invoke_config(
            effective_thread_id=effective_thread_id,
            recorder=recorder,
        )
        runtime_context = self._build_runtime_context(
            user_id=user_id,
            project_id=project_id,
            run_id=recorder.run_id,
            trace_id=recorder.trace_id,
        )

        return (
            runtime,
            recorder,
            effective_thread_id,
            invoke_config,
            runtime_context,
        )

    def run(
        self,
        query: str,
        *,
        project_id: int = 1,
        thread_id: str = "default",
        user_id: str = "local-user",
        project_root: str | None = None,
        chunks_path: str | None = None,
        index_path: str | None = None,
        recursion_limit: int | None = None,
        run_id: str | None = None,
        trace_id: str | None = None,
    ) -> LangChainAgentResult:
        (
            runtime,
            recorder,
            effective_thread_id,
            invoke_config,
            runtime_context,
        ) = self._prepare_run(
            query=query,
            project_id=project_id,
            thread_id=thread_id,
            user_id=user_id,
            project_root=project_root,
            chunks_path=chunks_path,
            index_path=index_path,
            recursion_limit=recursion_limit,
            run_id=run_id,
            trace_id=trace_id,
        )
        start_time = time.perf_counter()

        try:
            result = self._invoke_agent(
                agent=runtime.agent,
                query=query,
                invoke_config=invoke_config,
                runtime_context=runtime_context,
            )
        except GraphRecursionError as exc:
            return self._build_limited_result(
                query=query,
                project_id=project_id,
                thread_id=thread_id,
                effective_thread_id=effective_thread_id,
                recorder=recorder,
                start_time=start_time,
                error=exc,
            )

        messages = result.get("messages") if isinstance(result, dict) else None

        if not isinstance(messages, list):
            raise LangChainAgentExecutionError(
                "Agent 返回结果中缺少 messages 列表"
            )

        answer = _extract_final_answer(messages)
        tool_traces = _extract_tool_traces(messages)

        return self._build_result(
            query=query,
            project_id=project_id,
            thread_id=thread_id,
            effective_thread_id=effective_thread_id,
            recorder=recorder,
            start_time=start_time,
            messages=messages,
            answer=answer,
            tool_traces=tool_traces,
        )

    async def arun(
        self,
        *,
        query: str,
        project_id: int,
        thread_id: str,
        user_id: str = "local-user",
        project_root: str | None = None,
        chunks_path: str | None = None,
        index_path: str | None = None,
        recursion_limit: int | None = None,
        run_id: str | None = None,
        trace_id: str | None = None,
    ) -> LangChainAgentResult:
        (
            runtime,
            recorder,
            effective_thread_id,
            invoke_config,
            runtime_context,
        ) = self._prepare_run(
            query=query,
            project_id=project_id,
            thread_id=thread_id,
            user_id=user_id,
            project_root=project_root,
            chunks_path=chunks_path,
            index_path=index_path,
            recursion_limit=recursion_limit,
            run_id=run_id,
            trace_id=trace_id,
        )
        start_time = time.perf_counter()

        try:
            result = await self._ainvoke_agent(
                agent=runtime.agent,
                query=query,
                invoke_config=invoke_config,
                runtime_context=runtime_context,
            )
        except GraphRecursionError as exc:
            return self._build_limited_result(
                query=query,
                project_id=project_id,
                thread_id=thread_id,
                effective_thread_id=effective_thread_id,
                recorder=recorder,
                start_time=start_time,
                error=exc,
            )

        messages = result.get("messages") if isinstance(result, dict) else None

        if not isinstance(messages, list):
            raise LangChainAgentExecutionError(
                "Agent 返回结果中缺少 messages 列表"
            )

        answer = _extract_final_answer(messages)
        tool_traces = _extract_tool_traces(messages)

        return self._build_result(
            query=query,
            project_id=project_id,
            thread_id=thread_id,
            effective_thread_id=effective_thread_id,
            recorder=recorder,
            start_time=start_time,
            messages=messages,
            answer=answer,
            tool_traces=tool_traces,
        )
