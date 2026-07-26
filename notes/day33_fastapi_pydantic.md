# Day33 FastAPI 与 Pydantic 专项

## 1. FastAPI 请求体

`/langchain/agent` 使用 `LangChainAgentRequest` 作为请求体模型。

请求进入接口后，FastAPI 会先用 Pydantic 校验 JSON：

- 字段类型是否正确；
- 必填字段是否存在；
- `Field(ge=4, le=50)` 这类范围限制是否满足；
- 是否存在未声明字段。

如果校验失败，FastAPI 会直接返回 `422 Validation Error`。

## 2. 响应模型

`/langchain/agent` 使用 `LangChainAgentResult` 作为响应模型。

Day33 之后响应中增加：

- `run_id`
- `trace_id`
- `success`
- `degraded`
- `model_call_count`
- `tool_call_count`
- `message_trim_count`
- `trace`

这些字段用于观察一次 Agent 请求的执行过程。

## 3. ConfigDict(extra="forbid")

项目里的请求模型继承 `StrictRequestModel`：

```python
model_config = ConfigDict(extra="forbid")
```

作用是拒绝未声明字段。

这对 Agent 项目很重要，因为模型、前端或调用方传入多余参数时，不应该静默忽略，否则排查问题会很困难。

## 4. Field

`Field` 用来声明约束：

```python
recursion_limit: int = Field(default=20, ge=4, le=50)
```

含义：

- 默认值是 `20`
- 不能小于 `4`
- 不能大于 `50`

如果传入 `recursion_limit=0`，FastAPI 会返回 422。

## 5. SecretStr

`LangChainMiddlewareConfig` 中备用模型 API Key 使用 `SecretStr`。

它的意义不是加密，而是避免在日志、接口返回中直接泄露敏感信息。

对外展示时使用：

```python
fallback_api_key_configured: true / false
```

而不是返回真实 key。

## 6. Tool Schema 与 Tool Validation

Tool Schema：

```text
告诉模型工具需要哪些参数。
```

Tool Validation：

```text
工具真正执行前，再用 Pydantic 校验模型生成的参数。
```

所以它们分别面向：

- 模型生成参数；
- 后端安全执行。

## 7. 今天应该掌握

- FastAPI 会自动把请求 JSON 转成 Pydantic 模型；
- Pydantic 不只是类型提示，而是运行时校验；
- `extra="forbid"` 可以防止多余字段静默通过；
- `SecretStr` 可以避免 API Key 明文泄露；
- Tool Calling 里 Schema 给模型看，Validation 给后端兜底。
