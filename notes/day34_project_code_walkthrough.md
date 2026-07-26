# Day34 项目代码逐行复盘：短期记忆、Runtime Context 与异步 Agent

这份文档只复盘 Day34 主项目内容，不包含专项内容。

Day34 的核心目标是：让 `/langchain/agent` 不再只是一次性的工具调用 Agent，而是具备“同一个会话能记住前文、不同项目和不同会话互相隔离、API 层支持异步调用”的能力。

可以先记住这一条主线：

```text
HTTP 请求
→ LangChainAgentRequest 参数校验
→ LangChainAgentService.arun()
→ 构造 Runtime Context
→ 构造 effective_thread_id
→ 从 AgentRuntimeCache 获取 Agent + InMemorySaver
→ agent.ainvoke(..., config={"thread_id": ...}, context=...)
→ LangGraph 根据 thread_id 读写短期记忆
→ 返回答案、工具调用记录、Trace
```

---

## 1. `runtime_context.py`：描述一次 Agent 调用属于谁、哪个项目、读哪些文件

文件位置：

```text
app/langchain_agent/runtime_context.py
```

代码：

```python
from __future__ import annotations
```

这一行的作用是延迟解析类型注解。

好处是：类型注解里可以直接写当前还没完全加载的类名，减少循环引用和版本兼容问题。

```python
from dataclasses import dataclass, field
```

这里导入 Python 标准库里的 dataclass。

- `dataclass` 用来快速定义一个只保存数据的类；
- `field` 用来给字段设置更细的默认值逻辑。

```python
@dataclass(frozen=True)
class CodeDocRuntimeContext:
```

这里定义了 `CodeDocRuntimeContext`。

它是一个“运行时上下文对象”。

`frozen=True` 表示这个对象创建后不能随便改字段。

为什么这样设计？

因为这些字段代表后端可信上下文，例如用户 ID、项目 ID、项目路径、索引路径、权限信息。它们不应该在 Agent 执行过程中被随意修改。

```python
    user_id: str
```

表示当前请求属于哪个用户。

现在项目里默认是 `local-user`，后续如果接登录系统，这个字段就可以来自认证信息。

```python
    project_id: int
```

表示当前请求属于哪个项目。

它会参与生成 `effective_thread_id`，避免不同项目之间共享记忆。

```python
    project_root: str
```

表示当前项目根目录。

工具读取文件、扫描结构时，应该被限制在这个项目目录下。

```python
    chunks_path: str
```

表示当前项目的 chunk 文件路径。

比如：

```text
outputs/test_project_chunks.json
```

```python
    index_path: str
```

表示当前项目的向量索引路径。

比如：

```text
outputs/test_project_vector_index_bge_m3.json
```

```python
    run_id: str
    trace_id: str
```

这两个字段用于观测和追踪。

- `run_id`：一次 Agent 执行的 ID；
- `trace_id`：一次 Trace 链路的 ID。

它们不是业务回答内容，而是工程排查信息。

```python
    permissions: tuple[str, ...] = field(
        default_factory=lambda: ("project:read",)
    )
```

这里定义默认权限。

`default_factory` 的作用是：每次创建对象时生成一个新的默认值。

当前默认权限是：

```python
("project:read",)
```

意思是：当前 Agent 默认只有读取项目的权限。

这里用 tuple 而不是 list，是因为 tuple 更适合表达“不可变权限集合”。

```python
def build_effective_thread_id(
    *,
    project_id: int,
    thread_id: str,
) -> str:
```

这个函数专门负责生成真正传给 LangGraph 的线程 ID。

注意：用户传入的是普通 `thread_id`，例如：

```text
chat-001
```

项目内部真正使用的是：

```text
project:1:thread:chat-001
```

```python
    thread_id = thread_id.strip()
```

去掉用户传入 thread_id 两边的空格。

防止 `" chat-001 "` 和 `"chat-001"` 被当成两个不同会话。

```python
    if project_id <= 0:
        raise ValueError("project_id 必须大于 0")
```

校验项目 ID。

项目 ID 必须是正整数。

```python
    if not thread_id:
        raise ValueError("thread_id 不能为空")
```

校验 thread_id 不能为空。

如果允许空字符串，所有空 thread_id 的请求都会共用同一段记忆，会造成上下文污染。

```python
    return f"project:{project_id}:thread:{thread_id}"
```

生成最终线程 ID。

这是 Day34 非常关键的一行。

它解决的是“记忆隔离”问题。

如果只用：

```text
thread:chat-001
```

那么项目 1 和项目 2 可能串记忆。

现在变成：

```text
project:1:thread:chat-001
project:2:thread:chat-001
```

即使 thread_id 一样，也不会混在一起。

---

## 2. `checkpointer_factory.py`：创建 LangGraph 的短期记忆存储

文件位置：

```text
app/langchain_agent/checkpointer_factory.py
```

代码：

```python
from __future__ import annotations
```

同样是延迟解析类型注解。

```python
from langgraph.checkpoint.memory import InMemorySaver
```

这里导入 LangGraph 的内存版 checkpointer。

`InMemorySaver` 的作用是：在当前 Python 进程里保存 Agent 的中间状态和消息历史。

它是“短期记忆”，不是长期数据库。

特点：

- 进程内有效；
- 服务重启后丢失；
- 适合开发、测试、学习；
- 后续如果上生产，可以替换成 SQLite、Postgres、Redis 等持久化 checkpointer。

```python
def create_in_memory_checkpointer() -> InMemorySaver:
```

定义一个工厂函数。

为什么不直接在别的地方写 `InMemorySaver()`？

因为这样方便以后替换实现。

现在：

```python
return InMemorySaver()
```

以后可能变成：

```python
return SqliteSaver(...)
```

调用方不需要大改。

```python
    return InMemorySaver()
```

真正创建内存 checkpointer。

这就是 Day34 文档里说的：

```python
from langgraph.checkpoint.memory import InMemorySaver
```

在项目中的落点。

---

## 3. `agent_runtime.py`：把 Agent 和 Checkpointer 绑成一个运行时对象

文件位置：

```text
app/langchain_agent/agent_runtime.py
```

代码：

```python
from __future__ import annotations
```

延迟类型注解。

```python
from dataclasses import dataclass
from typing import Any
```

- `dataclass`：快速定义数据类；
- `Any`：表示 agent 类型比较复杂，不强行绑定具体类型。

```python
from langgraph.checkpoint.memory import InMemorySaver
```

这里导入 `InMemorySaver`，用于给 `checkpointer` 字段标注类型。

```python
@dataclass(slots=True)
class AgentRuntime:
```

定义 `AgentRuntime`。

`slots=True` 表示这个对象只能有声明过的字段，不能随便动态加字段。

好处：

- 更省内存；
- 字段更明确；
- 更像一个稳定的数据容器。

```python
    agent: Any
```

保存真正的 LangChain Agent。

也就是 `create_agent(...)` 返回的对象。

```python
    checkpointer: InMemorySaver
```

保存这个 Agent 对应的 checkpointer。

注意：短期记忆能不能持续，关键就是这个 checkpointer 是否被复用。

如果每次请求都新建一个 checkpointer，那么同一个 thread_id 也记不住前文。

```python
    tool_count: int
```

保存当前 Agent 有多少个工具。

这个字段主要用于返回结果和 Trace，例如告诉你本次 Agent 可用工具数量。

---

## 4. `agent_runtime_cache.py`：复用同一个 Agent + Checkpointer

文件位置：

```text
app/langchain_agent/agent_runtime_cache.py
```

这个文件是 Day34 里很关键的工程化部分。

如果没有它，`InMemorySaver` 可能每次请求都新建，短期记忆就会失效。

### 4.1 导入部分

```python
from __future__ import annotations
```

延迟类型注解。

```python
from pathlib import Path
```

用于把项目路径、chunks 路径、index 路径转换成绝对路径。

这样可以避免：

```text
outputs/chunks.json
.\outputs\chunks.json
D:\...\outputs\chunks.json
```

这些写法实际指向同一个文件，却被当成不同 key。

```python
from threading import Lock
```

导入线程锁。

因为 FastAPI 可能同时处理多个请求。

如果两个请求同时创建同一个项目的 runtime，没有锁就可能重复创建。

```python
from typing import Callable
```

用于标注 `create_agent_func` 是一个可调用函数。

```python
from langchain_core.tools import BaseTool
```

用于标注 tools 的类型。

```python
from langchain_agent.agent_runtime import AgentRuntime
from langchain_agent.checkpointer_factory import create_in_memory_checkpointer
from langchain_agent.model_config import LangChainModelConfig
```

导入项目内部类型：

- `AgentRuntime`：保存 agent、checkpointer、tool_count；
- `create_in_memory_checkpointer`：创建短期记忆；
- `LangChainModelConfig`：模型配置。

### 4.2 模型配置指纹

```python
def model_config_fingerprint(config: LangChainModelConfig) -> str:
```

这个函数用来给模型配置生成一个字符串指纹。

为什么要做模型配置指纹？

因为不同模型不应该共用同一个 Agent Runtime。

例如：

```text
qwen2.5:3b
qwen3.5:4b
```

它们的工具调用能力、上下文表现可能不同，不应该混在同一个 runtime 里。

```python
    return "|".join(
        [
            config.provider,
            config.model_name,
            config.base_url,
            str(config.temperature),
            str(config.max_tokens),
            str(config.max_retries),
        ]
    )
```

这里把关键模型配置拼成一个字符串。

例如：

```text
openai_compatible|qwen3.5:4b|http://localhost:11434/v1|0.1|300|0
```

只要其中一个配置变了，就会得到不同 fingerprint。

### 4.3 构造 Runtime Cache Key

```python
def build_project_runtime_key(
    *,
    project_id: int,
    project_root: str,
    chunks_path: str,
    index_path: str,
    model_config: LangChainModelConfig,
) -> tuple:
```

这个函数生成缓存 key。

它决定“什么情况下复用同一个 Agent Runtime”。

参数含义：

- `project_id`：项目 ID；
- `project_root`：项目根目录；
- `chunks_path`：chunk 文件；
- `index_path`：向量索引文件；
- `model_config`：模型配置。

```python
    root = str(Path(project_root).resolve())
    chunks = str(Path(chunks_path).resolve())
    index = str(Path(index_path).resolve())
```

把路径统一转成绝对路径。

这样缓存 key 更稳定。

```python
    return (
        project_id,
        root,
        chunks,
        index,
        model_config_fingerprint(model_config),
    )
```

返回一个 tuple 作为缓存 key。

也就是说：只有这些都一样，才复用同一个 runtime。

如果换了项目、换了索引、换了模型，都会创建新的 runtime。

### 4.4 AgentRuntimeCache 类

```python
class AgentRuntimeCache:
```

定义 Runtime 缓存类。

它在进程内保存：

```python
dict[tuple, AgentRuntime]
```

也就是：

```text
runtime_key → AgentRuntime
```

```python
    def __init__(self) -> None:
        self._lock = Lock()
        self._runtimes: dict[tuple, AgentRuntime] = {}
```

初始化两个字段：

- `_lock`：防止并发创建重复 runtime；
- `_runtimes`：真正保存缓存。

```python
    def clear(self) -> None:
        with self._lock:
            self._runtimes.clear()
```

清空缓存。

这个方法主要用于测试，或者后续你想手动重置 Agent 记忆。

```python
    def get_or_create(
        self,
        *,
        key: tuple,
        tools: list[BaseTool],
        create_agent_func: Callable[..., object],
    ) -> AgentRuntime:
```

这是核心方法。

语义是：如果 key 对应的 runtime 已经存在，就直接返回；否则创建一个新的。

```python
        with self._lock:
```

进入锁保护区域，避免并发问题。

```python
            existing = self._runtimes.get(key)
```

先查缓存。

```python
            if existing is not None:
                return existing
```

如果找到了，直接复用。

这就是短期记忆能够延续的关键。

```python
            checkpointer = create_in_memory_checkpointer()
```

如果没找到，就新建一个 `InMemorySaver`。

```python
            agent = create_agent_func(
                checkpointer=checkpointer,
            )
```

把 checkpointer 传给 Agent 创建函数。

这一步最终会走到：

```python
create_agent(..., checkpointer=checkpointer)
```

```python
            runtime = AgentRuntime(
                agent=agent,
                checkpointer=checkpointer,
                tool_count=len(tools),
            )
```

把 agent 和 checkpointer 绑成一个 `AgentRuntime`。

```python
            self._runtimes[key] = runtime
```

放入缓存。

```python
            return runtime
```

返回新创建的 runtime。

```python
GLOBAL_AGENT_RUNTIME_CACHE = AgentRuntimeCache()
```

定义一个全局 Runtime Cache。

API 请求默认都会复用这个全局缓存。

所以同一个 Python 服务进程里，同一个项目、同一个索引、同一个模型配置，可以保留短期记忆。

---

## 5. `agent_factory.py`：真正把 checkpointer 接入 create_agent

文件位置：

```text
app/langchain_agent/agent_factory.py
```

这里是 Day34 文档中这段代码的实际落点：

```python
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    middleware=middleware,
    context_schema=CodeDocRuntimeContext,
    checkpointer=checkpointer,
    name="codedoc_agent",
)
```

### 5.1 导入部分

```python
from langchain.agents import create_agent
```

导入 LangChain v1 的 Agent 创建入口。

这是现代 LangChain 的主线能力之一。

```python
from langchain_core.tools import BaseTool
```

用于给 tools 做类型标注。

```python
from langchain_agent.model_config import LangChainModelConfig
from langchain_agent.model_factory import create_chat_model
from langchain_agent.runtime_context import CodeDocRuntimeContext
```

导入：

- 模型配置；
- 模型创建函数；
- Day34 新增的 Runtime Context。

### 5.2 系统提示词

```python
CODEDOC_AGENT_SYSTEM_PROMPT = """...""".strip()
```

这是 Agent 的系统提示词。

它告诉模型：

- 你是 CodeDoc Research Agent；
- 只能根据项目真实代码、文档、目录结构回答；
- 可以使用哪些工具；
- 不要编造文件、函数和实现。

注意：用户之前不希望加过度约束，所以这里保留的是基础工具使用规则，没有写死某个问题必须调用某几个工具。

### 5.3 配置错误类型

```python
class LangChainAgentConfigurationError(ValueError):
    pass
```

定义一个专门的配置异常。

当模型配置不支持工具调用时，会抛这个异常。

API 层会把它转成 400 错误。

### 5.4 创建 Agent

```python
def create_codedoc_agent(
    *,
    config: LangChainModelConfig,
    tools: list[BaseTool],
    middleware: list[Any] | None = None,
    checkpointer: Any | None = None,
) -> Any:
```

这是项目里的 Agent 工厂函数。

参数含义：

- `config`：模型配置；
- `tools`：LangChain Tool 列表；
- `middleware`：Day33 的中间件能力预留；
- `checkpointer`：Day34 新增，用于短期记忆。

```python
    if config.provider == "mock":
        raise LangChainAgentConfigurationError(...)
```

如果 provider 是 mock，就不允许创建 Agent。

原因：mock 模型不具备真实 tool calling 能力。

```python
    model = create_chat_model(config)
```

根据配置创建真实 ChatModel。

比如你本地常用的是：

```text
openai_compatible + Ollama + qwen3.5:4b
```

```python
    kwargs = {
        "model": model,
        "tools": tools,
        "system_prompt": CODEDOC_AGENT_SYSTEM_PROMPT,
        "context_schema": CodeDocRuntimeContext,
        "name": "codedoc_agent",
    }
```

这里先把创建 Agent 的参数放到字典里。

逐个看：

```python
"model": model
```

Agent 使用哪个大模型。

```python
"tools": tools
```

Agent 可以调用哪些工具。

比如：

- `search_code`
- `search_documents`
- `get_project_structure`
- `get_symbol_definition`
- `read_file_range`

```python
"system_prompt": CODEDOC_AGENT_SYSTEM_PROMPT
```

Agent 的系统提示词。

```python
"context_schema": CodeDocRuntimeContext
```

这是 Day34 新增重点。

它告诉 LangChain：本 Agent 运行时可以接收 `CodeDocRuntimeContext` 这种上下文对象。

```python
"name": "codedoc_agent"
```

给 Agent 命名。

后续 Trace 或调试时更容易识别。

```python
    if middleware:
        kwargs["middleware"] = middleware
```

如果传入 middleware，就加入 Agent 创建参数。

```python
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
```

这是 Day34 最关键的接入点。

只有把 `checkpointer` 传进 `create_agent`，LangGraph 才能根据 `thread_id` 保存和恢复短期记忆。

```python
    try:
        return create_agent(**kwargs)
```

真正创建 LangChain Agent。

```python
    except TypeError:
        kwargs.pop("middleware", None)
        kwargs.pop("context_schema", None)
        kwargs.pop("checkpointer", None)
        return create_agent(**kwargs)
```

这是兼容逻辑。

如果当前安装的 LangChain 版本不支持某些参数，就去掉这些参数后再创建。

但你当前环境已经确认支持：

- `context_schema`
- `checkpointer`

所以正常情况下会走第一条 `create_agent(**kwargs)`。

---

## 6. `agent_service.py`：把 API 请求变成一次带记忆的 Agent 调用

文件位置：

```text
app/langchain_agent/agent_service.py
```

这个文件是 Day34 链路最核心的调度层。

### 6.1 `_build_runtime()`：拿到 Agent + Checkpointer

```python
def _build_runtime(
    self,
    *,
    project_id: int,
) -> AgentRuntime:
```

这个函数根据项目 ID 获取当前要用的 Agent Runtime。

```python
    if self._agent is not None:
```

这是测试用分支。

如果测试里传了 fake agent，就不创建真实 LangChain Agent。

```python
        checkpointer = create_in_memory_checkpointer()
        return AgentRuntime(
            agent=self._agent,
            checkpointer=checkpointer,
            tool_count=self._tool_count,
        )
```

为测试 agent 包一层 runtime。

真实运行时一般不会走这个分支。

```python
    tools = self._build_tools()
```

构建工具列表。

这里会把项目里的工具注册表转换成 LangChain Tools。

```python
    key = build_project_runtime_key(
        project_id=project_id,
        project_root=self.project_root,
        chunks_path=self.chunks_path,
        index_path=self.index_path,
        model_config=self.config,
    )
```

生成 runtime cache key。

同一个项目、同一个索引、同一个模型配置会得到同一个 key。

```python
    runtime = self.runtime_cache.get_or_create(
        key=key,
        tools=tools,
        create_agent_func=lambda checkpointer: create_codedoc_agent(
            config=self.config,
            tools=tools,
            checkpointer=checkpointer,
        ),
    )
```

这里发生了三件事：

1. 去缓存里找 runtime；
2. 如果没有，就创建 `InMemorySaver`；
3. 把 checkpointer 传给 `create_codedoc_agent()`。

最终会进入：

```python
create_agent(..., checkpointer=checkpointer)
```

```python
    self._tool_count = runtime.tool_count
    return runtime
```

保存工具数量，然后返回 runtime。

### 6.2 `_build_invoke_config()`：生成 LangGraph 的 thread_id 配置

```python
def _build_invoke_config(
    self,
    *,
    effective_thread_id: str,
    recorder: AgentTraceRecorder,
) -> dict:
```

这个函数生成传给 Agent 的 `config`。

```python
    return {
        "recursion_limit": self.recursion_limit,
```

限制 Agent 图执行的最大递归步数。

这可以防止工具调用死循环。

```python
        "configurable": {
            "thread_id": effective_thread_id,
        },
```

这是 Day34 最关键的调用配置。

LangGraph 的 checkpointer 会根据这里的 `thread_id` 保存和恢复状态。

如果两次请求传入同一个：

```text
project:1:thread:chat-001
```

就会接上同一段短期记忆。

如果 thread_id 不同，就互相隔离。

```python
        "metadata": {
            "run_id": recorder.run_id,
            "trace_id": recorder.trace_id,
        },
```

metadata 用于 Trace 和调试。

它不决定记忆隔离，真正决定记忆隔离的是 `configurable.thread_id`。

### 6.3 `_build_runtime_context()`：创建 Runtime Context

```python
def _build_runtime_context(
    self,
    *,
    user_id: str,
    project_id: int,
    run_id: str,
    trace_id: str,
) -> CodeDocRuntimeContext:
```

这个函数把本次请求的用户、项目、路径和 trace 信息打包成上下文。

```python
    return CodeDocRuntimeContext(
        user_id=user_id,
        project_id=project_id,
        project_root=self.project_root,
        chunks_path=self.chunks_path,
        index_path=self.index_path,
        run_id=run_id,
        trace_id=trace_id,
    )
```

返回一个不可变的上下文对象。

这份上下文会在调用 Agent 时传入：

```python
context=runtime_context
```

### 6.4 `_invoke_agent()`：同步调用 Agent

```python
def _invoke_agent(
    self,
    *,
    agent: Any,
    query: str,
    invoke_config: dict,
    runtime_context: CodeDocRuntimeContext,
) -> dict:
```

这是同步调用版本。

```python
    payload = {
        "messages": [
            HumanMessage(content=query),
        ]
    }
```

把用户问题包装成 LangChain 消息格式。

Day34 文档里写的是：

```python
{
    "messages": [
        {
            "role": "user",
            "content": query,
        }
    ]
}
```

项目里用的是：

```python
HumanMessage(content=query)
```

两者语义一样。

项目里使用 LangChain 的消息对象，更类型安全。

```python
    if _supports_parameter(agent.invoke, "context"):
```

检查当前 agent 的 `invoke` 方法是否支持 `context` 参数。

这是为了兼容测试 fake agent 或不同 LangChain 版本。

```python
        return agent.invoke(
            payload,
            config=invoke_config,
            context=runtime_context,
        )
```

这是 Day34 文档里调用方式的真实对应代码。

关键点：

- `payload` 里放用户消息；
- `config` 里放 `thread_id`；
- `context` 里放 Runtime Context。

```python
    return agent.invoke(
        payload,
        config=invoke_config,
    )
```

如果 agent 不支持 context，就退化成不传 context 的调用。

### 6.5 `_ainvoke_agent()`：异步调用 Agent

```python
async def _ainvoke_agent(
```

这是异步版本。

FastAPI 的 `/langchain/agent` 最终走的是这个异步链路。

```python
    payload = {
        "messages": [
            HumanMessage(content=query),
        ]
    }
```

同样构造用户消息。

```python
    if hasattr(agent, "ainvoke"):
```

如果 agent 支持原生异步调用，就走 `ainvoke`。

```python
        if _supports_parameter(agent.ainvoke, "context"):
            return await agent.ainvoke(
                payload,
                config=invoke_config,
                context=runtime_context,
            )
```

异步版本也会传：

- `config`
- `context`

所以短期记忆和 Runtime Context 在 async 场景下同样生效。

```python
        return await agent.ainvoke(
            payload,
            config=invoke_config,
        )
```

如果不支持 context，就只传 config。

```python
    return await asyncio.to_thread(
        self._invoke_agent,
        agent=agent,
        query=query,
        invoke_config=invoke_config,
        runtime_context=runtime_context,
    )
```

如果 agent 没有 `ainvoke`，就把同步调用丢到线程里执行。

这样 API 层仍然可以保持 async。

### 6.6 `_prepare_run()`：把一次请求准备成 Agent 可执行状态

```python
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
```

这个函数是 Day34 调用前的总准备入口。

它返回五个东西：

```text
runtime
recorder
effective_thread_id
invoke_config
runtime_context
```

```python
    query = query.strip()
```

去掉 query 两边空格。

```python
    if not query:
        raise ValueError("query 不能为空")
```

防止空问题进入 Agent。

```python
    if project_root is not None:
        self.project_root = project_root
```

如果 API 请求传了项目根目录，就覆盖 service 默认值。

下面几个类似：

```python
    if chunks_path is not None:
        self.chunks_path = chunks_path

    if index_path is not None:
        self.index_path = index_path

    if recursion_limit is not None:
        self.recursion_limit = recursion_limit
```

分别更新 chunks 路径、index 路径、递归限制。

```python
    self._validate_paths()
```

校验项目路径和 chunks 文件是否存在。

```python
    recorder = AgentTraceRecorder(
        run_id=run_id,
        trace_id=trace_id,
    )
```

创建 Trace 记录器。

```python
    middleware_bundle = build_agent_middleware(
        config=self.middleware_config,
        recorder=recorder,
        primary_model_config=self.config,
    )
    middleware_bundle.fallback.mark_if_enabled()
```

保留 Day33 的中间件和降级标记能力。

Day34 没有重写这部分，只是在它基础上继续加记忆和上下文。

```python
    runtime = self._build_runtime(
        project_id=project_id,
    )
```

获取 Agent Runtime。

这里会拿到：

- agent；
- checkpointer；
- tool_count。

```python
    effective_thread_id = resolve_effective_thread_id(
        project_id=project_id,
        thread_id=thread_id,
    )
```

生成隔离后的真实 thread_id。

例如：

```text
project:1:thread:chat-001
```

```python
    invoke_config = self._build_invoke_config(
        effective_thread_id=effective_thread_id,
        recorder=recorder,
    )
```

生成传给 Agent 的 config。

其中最重要的是：

```python
configurable.thread_id
```

```python
    runtime_context = self._build_runtime_context(
        user_id=user_id,
        project_id=project_id,
        run_id=recorder.run_id,
        trace_id=recorder.trace_id,
    )
```

创建 Runtime Context。

```python
    return (
        runtime,
        recorder,
        effective_thread_id,
        invoke_config,
        runtime_context,
    )
```

返回调用 Agent 所需的所有对象。

### 6.7 `run()`：同步入口

```python
def run(
    self,
    query: str,
    *,
    project_id: int = 1,
    thread_id: str = "default",
    user_id: str = "local-user",
    ...
) -> LangChainAgentResult:
```

这是同步调用入口。

默认：

- `project_id=1`
- `thread_id="default"`
- `user_id="local-user"`

所以旧测试和旧调用不传这些字段也能继续跑。

```python
    (
        runtime,
        recorder,
        effective_thread_id,
        invoke_config,
        runtime_context,
    ) = self._prepare_run(...)
```

先准备运行时对象。

```python
    start_time = time.perf_counter()
```

记录开始时间，用于统计耗时。

```python
    try:
        result = self._invoke_agent(...)
```

真正调用 Agent。

```python
    except GraphRecursionError as exc:
        return self._build_limited_result(...)
```

如果 Agent 超过递归限制，就返回受限结果，避免服务崩溃。

```python
    messages = result.get("messages") if isinstance(result, dict) else None
```

从 Agent 返回结果中拿 messages。

LangChain Agent 的核心输出就是消息列表。

```python
    if not isinstance(messages, list):
        raise LangChainAgentExecutionError(...)
```

如果没有 messages，说明 Agent 返回格式异常。

```python
    answer = _extract_final_answer(messages)
```

从最后一条 AIMessage 中提取最终答案。

```python
    tool_traces = _extract_tool_traces(messages)
```

从消息列表里解析工具调用记录。

```python
    return self._build_result(...)
```

构造统一响应对象。

### 6.8 `arun()`：异步入口

```python
async def arun(
```

这是 Day34 新增重点之一。

API 层现在使用它。

```python
    (
        runtime,
        recorder,
        effective_thread_id,
        invoke_config,
        runtime_context,
    ) = self._prepare_run(...)
```

异步入口和同步入口一样，也先准备运行时。

```python
    result = await self._ainvoke_agent(
        agent=runtime.agent,
        query=query,
        invoke_config=invoke_config,
        runtime_context=runtime_context,
    )
```

调用异步 Agent。

这一步对应：

```python
await agent.ainvoke(..., config=..., context=...)
```

```python
    answer = _extract_final_answer(messages)
    tool_traces = _extract_tool_traces(messages)
```

同样提取答案和工具轨迹。

```python
    return self._build_result(...)
```

返回统一结构。

---

## 7. `langchain_schema.py`：让 API 请求支持项目、用户和会话字段

文件位置：

```text
app/schemas/langchain_schema.py
```

Day34 修改重点在 `LangChainAgentRequest`。

```python
class LangChainAgentRequest(StrictRequestModel):
```

这是 `/langchain/agent` 的请求体 Schema。

```python
    query: str = Field(
        min_length=1,
        max_length=3000,
    )
```

用户问题。

不能为空，最多 3000 字符。

```python
    project_id: int = Field(
        default=1,
        ge=1,
    )
```

Day34 新增字段。

表示当前请求属于哪个项目。

默认是 1，方便本地测试。

```python
    thread_id: str = Field(
        default="default",
        min_length=1,
        max_length=100,
    )
```

Day34 新增字段。

表示当前对话线程。

同一个 `project_id + thread_id` 会共享短期记忆。

```python
    user_id: str = Field(
        default="local-user",
        min_length=1,
        max_length=100,
    )
```

Day34 新增字段。

表示当前用户。

现在主要进入 Runtime Context，后续可以用于权限控制和用户隔离。

```python
    project_root: str = Field(...)
    chunks_path: str = Field(...)
    index_path: str = Field(...)
```

这些字段告诉 Agent 工具应该操作哪个项目、哪个 chunks 文件、哪个向量索引。

```python
    recursion_limit: int = Field(
        default=20,
        ge=4,
        le=50,
    )
```

限制 Agent 最大执行步数。

防止模型不停调用工具。

后面的 embedding 和 rerank 字段用于工具检索链路，不是 Day34 新增核心，但 `/langchain/agent` 需要它们来构建工具。

---

## 8. `langchain_router.py`：API 层改成 async 并传入 Day34 字段

文件位置：

```text
app/api/langchain_router.py
```

Day34 重点是 `/langchain/agent`。

```python
@router.post(
    "/agent",
    response_model=LangChainAgentResult,
)
async def run_langchain_agent(
    request: LangChainAgentRequest,
) -> LangChainAgentResult:
```

这里把接口函数改成了 `async def`。

为什么？

因为 Agent 调用模型、工具、检索时可能比较耗时，异步接口更适合 Web 服务。

```python
    config = LangChainModelConfig.from_env()
```

从环境变量读取 LangChain 模型配置。

例如：

```powershell
$env:LANGCHAIN_CHAT_MODEL="qwen3.5:4b"
$env:LANGCHAIN_CHAT_BASE_URL="http://localhost:11434/v1"
```

```python
    middleware_config = LangChainMiddlewareConfig.from_env()
```

读取 Day33 的中间件配置。

```python
    service = LangChainAgentService(
        config=config,
        middleware_config=middleware_config,
        project_root=request.project_root,
        chunks_path=request.chunks_path,
        index_path=request.index_path,
        recursion_limit=request.recursion_limit,
        ...
    )
```

构造 Agent Service。

注意：这里把请求体里的项目路径、chunks、index、embedding、rerank 参数都传给 service。

```python
        return await service.arun(
            query=request.query,
            project_id=request.project_id,
            thread_id=request.thread_id,
            user_id=request.user_id,
            project_root=request.project_root,
            chunks_path=request.chunks_path,
            index_path=request.index_path,
            recursion_limit=request.recursion_limit,
            run_id=request.run_id,
            trace_id=request.trace_id,
        )
```

这是 API 层和 Day34 Service 层的连接点。

关键字段：

- `project_id`
- `thread_id`
- `user_id`

会进入：

```text
CodeDocRuntimeContext
effective_thread_id
agent.ainvoke config
```

---

## 9. Day34 新增测试怎么证明功能没问题

### 9.1 `test_agent_runtime_cache.py`

验证 Runtime Cache：

- 相同 key 复用同一个 runtime；
- 不同 key 创建不同 runtime；
- model config 变化时 key 也变化。

它证明：短期记忆的 checkpointer 不会每次请求都丢失。

### 9.2 `test_langchain_short_term_memory.py`

验证同一个线程能记住前文。

测试逻辑大概是：

```text
第一次问：项目使用的重排客户端叫 RerankClient
第二次问：它在哪个文件？
```

如果同一个 thread_id 能记住前文，第二次就知道“它”指的是 `RerankClient`。

### 9.3 `test_langchain_thread_isolation.py`

验证隔离：

- thread-A 里说过 RerankClient；
- thread-B 里再问“它在哪”；
- thread-B 不应该知道“它”是谁。

还验证：

- project 1 的 shared-thread；
- project 2 的 shared-thread；

即使 thread_id 一样，只要 project_id 不一样，也不能串记忆。

### 9.4 `test_langchain_runtime_context.py`

验证 Runtime Context：

- `CodeDocRuntimeContext` 默认权限是 `("project:read",)`；
- `build_effective_thread_id()` 生成格式正确；
- Runtime Context 字段不会暴露到 Tool Schema 里。

最后一点很重要。

模型可以看到工具参数 Schema，但不应该让模型自己传 `user_id`、`project_id`、`permissions` 这种后端可信字段。

### 9.5 `test_langchain_agent_async.py`

验证异步 Agent 调用：

- `service.arun()` 能正常执行；
- `context` 能传给 agent；
- `thread_id` 能进入 config；
- 返回结果里有 project/thread 信息。

---

## 10. Day34 的核心知识点

### 10.1 `InMemorySaver` 是什么

它是 LangGraph 的内存版 checkpointer。

它负责保存 Agent 的状态和消息历史。

没有 checkpointer：

```text
每次请求都是全新对话
```

有 checkpointer + thread_id：

```text
同一个 thread_id 可以延续对话
```

### 10.2 `thread_id` 是什么

`thread_id` 是 LangGraph 识别一段对话记忆的 key。

你的项目没有直接用用户传入的 thread_id，而是构造：

```text
project:{project_id}:thread:{thread_id}
```

这是更合理的工程设计。

### 10.3 Runtime Context 是什么

Runtime Context 是后端传给 Agent 的可信上下文。

它不等于 prompt。

prompt 是给模型看的自然语言指令。

Runtime Context 是后端运行时信息。

例如：

```text
当前用户是谁
当前项目是谁
当前索引文件在哪里
当前权限是什么
当前 trace_id 是什么
```

这些东西不应该让模型自己生成。

### 10.4 Runtime Cache 是什么

Runtime Cache 负责复用 Agent 和 Checkpointer。

没有 Runtime Cache，即使你传同一个 thread_id，也可能因为 checkpointer 每次重建而记不住。

Day34 的关键不是单独用了 `InMemorySaver`，而是：

```text
InMemorySaver + thread_id + Runtime Cache
```

这三个一起工作，短期记忆才成立。

### 10.5 async API 的意义

FastAPI 中：

```python
async def run_langchain_agent(...)
```

配合：

```python
await service.arun(...)
```

表示接口可以异步等待模型和工具调用。

这更接近真实后端项目。

---

## 11. Day34 你应该怎么从头串起来

你可以这样理解：

用户请求：

```json
{
  "query": "它在哪个文件？",
  "project_id": 1,
  "thread_id": "chat-001",
  "user_id": "local-user"
}
```

进入 API：

```text
langchain_router.py
```

API 创建：

```text
LangChainAgentService
```

Service 准备：

```text
_prepare_run()
```

里面做：

```text
校验 query
校验路径
创建 recorder
获取 runtime
生成 effective_thread_id
生成 invoke_config
生成 runtime_context
```

然后调用：

```text
_ainvoke_agent()
```

实际传入 Agent：

```python
payload = {
    "messages": [
        HumanMessage(content=query)
    ]
}
```

```python
config = {
    "configurable": {
        "thread_id": "project:1:thread:chat-001"
    }
}
```

```python
context = CodeDocRuntimeContext(...)
```

Agent 执行时，LangGraph 根据：

```text
thread_id
```

去 InMemorySaver 里找历史消息。

如果有，就接着之前的对话继续。

如果没有，就从新对话开始。

---

## 12. 今天代码的验收标准

你看 Day34 代码时，可以用这几个问题自测：

1. `InMemorySaver` 在哪里创建？

答案：

```text
checkpointer_factory.py
```

2. `checkpointer` 在哪里传给 `create_agent`？

答案：

```text
agent_factory.py 的 create_codedoc_agent()
```

3. `thread_id` 在哪里变成 `project:{project_id}:thread:{thread_id}`？

答案：

```text
runtime_context.py / conversation_service.py
```

4. `thread_id` 最终在哪里传给 Agent？

答案：

```text
agent_service.py 的 _build_invoke_config()
```

5. Runtime Context 在哪里创建？

答案：

```text
agent_service.py 的 _build_runtime_context()
```

6. Runtime Context 在哪里传给 Agent？

答案：

```text
agent_service.py 的 _invoke_agent() 和 _ainvoke_agent()
```

7. API 层在哪里调用异步 Agent？

答案：

```text
langchain_router.py 的 run_langchain_agent()
```

8. 为什么需要 Runtime Cache？

答案：

```text
为了复用同一个 Agent + Checkpointer，否则短期记忆会随着每次新建而丢失。
```

---

## 13. Day34 推荐测试命令

只测 Day34 相关内容：

```powershell
python -m pytest tests\test_agent_runtime_cache.py tests\test_langchain_short_term_memory.py tests\test_langchain_thread_isolation.py tests\test_langchain_runtime_context.py tests\test_langchain_agent_async.py tests\test_langchain_agent_service.py tests\test_langchain_schema.py tests\test_langchain_router.py -q --basetemp .tmp_pytest_day34
```

全量测试：

```powershell
python -m pytest -q --basetemp .tmp_pytest_full_day34
```

当前已验收结果：

```text
364 passed
```

---

## 14. 一句话总结 Day34

Day34 不是简单“加了一个 InMemorySaver”。

真正完成的是：

```text
让 LangChain Agent 从一次性工具调用，升级成按项目和会话隔离的多轮 Agent。
```

它的工程关键词是：

```text
InMemorySaver
thread_id
Runtime Context
Runtime Cache
async Agent API
conversation isolation
```

这也是后面进入 LangGraph 的前置基础。
