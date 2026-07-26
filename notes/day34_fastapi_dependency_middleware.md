# Day34 FastAPI Dependency 与 Middleware 复盘

## 1. Dependency 适合做什么

FastAPI Dependency 更适合处理“请求进入接口之前需要准备好的对象或配置”，例如：

- 读取环境变量；
- 构造 Service；
- 校验公共参数；
- 获取当前用户；
- 注入数据库连接、模型配置或运行时配置。

在当前项目里，`/langchain/agent` 接口会先从请求体里拿到 `project_id`、`thread_id`、`user_id`、`project_root`、`chunks_path`、`index_path` 等信息，再交给 `LangChainAgentService`。

## 2. Middleware 适合做什么

Middleware 更适合处理“所有请求都要经过的一层横切逻辑”，例如：

- 统一日志；
- 统计请求耗时；
- 统一异常处理；
- CORS；
- 鉴权入口；
- Trace ID 注入。

它不适合写太多业务逻辑。比如“这个 Agent 应该读取哪个项目的索引文件”，就不应该放在 Middleware 里，而应该放在 Service 或 Runtime Context 里。

## 3. Day34 在项目里的对应关系

Day34 的重点不是把 FastAPI Middleware 写复杂，而是把 Agent 运行时信息从接口层传入 LangChain Agent：

```text
HTTP Request
→ LangChainAgentRequest
→ LangChainAgentService.arun
→ CodeDocRuntimeContext
→ LangChain create_agent runtime context
→ Tool / Model 执行
```

## 4. 为什么要有 Runtime Context

Runtime Context 用来保存“这次 Agent 运行属于谁、属于哪个项目、使用哪个线程、读取哪些索引文件”。

这些信息不应该让模型自由生成，也不应该混进用户 prompt 里。

更合理的方式是：

- 用户问题放在 messages；
- 项目和权限信息放在 runtime context；
- `thread_id` 放在 configurable；
- trace 信息放在 metadata。

这样工程边界更清楚，也更接近真实后端系统的设计。

## 5. 今天应该掌握的点

- FastAPI Dependency 偏“对象注入和参数准备”；
- Middleware 偏“统一横切逻辑”；
- Runtime Context 偏“本次 Agent 运行的业务上下文”；
- `thread_id` 是短期记忆隔离的关键；
- `project_id + thread_id` 可以防止不同项目之间串记忆。
