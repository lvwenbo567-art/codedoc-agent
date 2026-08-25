# CodeDoc 本地 Demo 指南

本指南使用仓库内的 `test_project`，不需要上传外部代码即可验证完整主链路。

## 1. 启动服务

优先参考项目根目录 [README](../README.md) 的 Docker Compose 或本地启动命令。服务启动后访问：

- 前端：`http://127.0.0.1:5173`
- 后端 Swagger：`http://127.0.0.1:8000/docs`

> 使用真实检索和 Tool Calling 前，需要保证本机 Ollama 已运行并具备 `bge-m3`、`qwen3.5:4b` 模型；默认 Reranker 路径也需要按本机环境配置。

## 2. 选择项目并构建知识库

1. 在前端项目工作区选择 `test_project`，或上传一个 ZIP 项目。
2. 点击“扫描并生成 chunks”。Python 文件会按 AST 切分，文档和配置文件会按结构切分。
3. 点击“构建向量索引”，使用 `bge-m3` 为 chunk 写入向量。
4. 确认工作区展示了 chunks 与索引路径，再进入问答页。

## 3. 推荐演示问题

### A. 精确代码定位

```text
keyword_score 函数在哪里定义？它的作用是什么？
```

观察点：Agent 应优先调用符号定位；答案应包含 `test_project/search.py`、函数名和实现含义。

### B. 文档证据问答

```text
README 或使用文档里有没有说明这个项目怎么启动？
```

观察点：Agent 应调用文档检索；回答应基于 `README` 或 `docs/usage.md`，而不是脱离项目内容生成。

### C. 项目导航

```text
这个项目有哪些主要目录和模块？请先查看项目结构再回答。
```

观察点：Agent 调用项目结构工具，并在答案中标出目录、模块与可能的截断状态。

### D. 需要人工确认的测试执行

```text
请运行 tests/test_project_test_tools.py，验证 run_project_tests 工具是否正常。
```

观察点：前端收到 `tool_approval` 中断事件，展示 `run_project_tests` 的工具参数。选择：

- `approve`：继续执行测试并让 Agent 汇总结果；
- `reject`：拒绝本次执行，图安全停止；
- `edit`：在保持 tool_call id 不变的前提下修改允许编辑的参数后继续。

## 4. 观察 Trace 与反馈闭环

一次执行完成后，前端会保留执行步骤、工具调用记录、引用证据和最终状态。对回答提交反馈后，可通过评测页或脚本将失败样本沉淀为 Bad Case。

Trace 重点用于定位：

1. 检索是否没有召回正确证据；
2. Agent 是否选择了错误或额外工具；
3. 工具是否超时/失败；
4. 模型是否忽略证据或遗漏答案关键点。
