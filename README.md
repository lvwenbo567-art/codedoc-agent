# CodeDoc Research Agent

一个面向代码仓库理解与研发问答场景的 Agentic RAG 系统。项目支持代码仓库解析、代码与文档联合检索、引用式问答、工具调用、多轮上下文管理、人工审核和自动化评测。

> 当前项目用于学习和展示大模型应用工程能力，重点覆盖 RAG、Function Calling、LangChain Tools、LangGraph 工作流、Checkpoint 记忆、HITL 和评测闭环。

## 核心能力

- 代码仓库接入：扫描项目文件，解析 Python AST，切分函数、类、方法、Markdown 文档和 JSON/YAML/TOML 配置。
- 检索链路：支持 BM25、向量检索、Hybrid Search、Query Rewrite、Multi-Query 和 Reranker。
- 存储后端：支持本地 JSON 向量索引和 Qdrant 向量数据库。
- RAG 问答：基于检索证据构造上下文，生成引用，校验 Citation，并在证据不足时拒答。
- Agent 工具：封装代码搜索、文档检索、项目结构查询、符号定位、源码读取、测试运行等工具。
- LangGraph 工作流：支持工具路由、证据评估、条件分支、安全停止、Checkpoint、会话摘要和长期记忆。
- Human-in-the-loop：对测试运行等需要确认的工具调用支持 approve、reject、edit。
- 自动化评测：提供 JSONL 评测集和脚本，统计 Recall@K、MRR、NDCG、工具选择准确率、任务成功率和延迟。

## 项目结构

```text
app/
  api/                  FastAPI 路由
  ingestion/            文件扫描、代码解析、chunk 构建
  services/             检索、索引、RAG、异步 Job 等业务服务
  vectorstores/         JSON / Qdrant 向量存储实现
  tools/                自研工具注册、执行与工具函数
  function_calling/     手写 Function Calling 循环
  langchain_agent/      LangChain 模型、工具和中间件集成
  langgraph_agent/      LangGraph Agent、HITL、Checkpoint、SSE
  memory/               结构化长期记忆与会话摘要
  context_engineering/  上下文预算、证据选择、安全上下文
  security/             Prompt Injection 检测与敏感信息脱敏
  evaluation/           检索与 Agent 评测
  mcp/                  MCP 只读适配
  skills/               Agent Skills
frontend/               React + Vite 前端工作台
tests/                  自动化测试
scripts/                实验与运维脚本
data/evaluation/        JSONL 评测集
outputs/experiments/    可保留的实验报告
examples/               示例项目
```

更多后端目录说明见 [app/README.md](app/README.md)。

## 快速启动

如果希望使用 Docker Compose 一次性启动 FastAPI、Qdrant、Redis、Ollama 和前端，请参考 [Docker Compose 部署说明](docs/docker_compose_deployment.md)。

### 1. 安装 Python 依赖

```powershell
cd codedoc-agent
pip install -r requirements.txt
```

### 2. 准备 Ollama 模型

```powershell
ollama pull qwen3.5:4b
ollama pull bge-m3
ollama list
```

### 3. 启动后端

```powershell
$env:LANGCHAIN_CHAT_PROVIDER="openai_compatible"
$env:LANGCHAIN_CHAT_MODEL="qwen3.5:4b"
$env:LANGCHAIN_CHAT_BASE_URL="http://localhost:11434/v1"
$env:LANGCHAIN_CHAT_API_KEY="EMPTY"
$env:LANGCHAIN_CHAT_TIMEOUT_SECONDS="180"
$env:LANGCHAIN_CHAT_MAX_TOKENS="600"
$env:LANGCHAIN_CHAT_MAX_RETRIES="0"
$env:LANGCHAIN_CHAT_TEMPERATURE="0.1"

uvicorn api_main:app --reload --app-dir app
```

访问：

```text
http://127.0.0.1:8000/docs
```

### 4. 启动前端

```powershell
cd frontend
npm install
npm run dev
```

访问：

```text
http://127.0.0.1:5173
```

前端会通过 Vite 代理请求本地 FastAPI 后端。

## 基础使用流程

1. 在前端上传 ZIP 项目，或使用 `test_project` 示例项目。
2. 点击“扫描”，生成代码和文档 chunks。
3. 点击“构建索引”，生成向量索引。
4. 进入“问答”，围绕当前仓库提问。
5. 对需要确认的工具调用进行 approve / reject / edit。
6. 在“评测”页导入实验报告，或对回答提交反馈并沉淀 Bad Case。

推荐测试问题：

```text
keyword_score 在哪里定义？
项目有哪些主要模块？
README 里怎么启动项目？
运行 tests/test_search.py
```

## 自动化测试

运行核心测试：

```powershell
python -m pytest
```

只验证 MCP、Skills、工具调用和检索策略相关测试：

```powershell
python -m pytest `
  tests/test_api_mcp.py `
  tests/test_api_skills.py `
  tests/test_mcp_adapter.py `
  tests/test_skills.py `
  tests/test_tool_call_normalizer.py `
  tests/test_project_test_tools.py `
  tests/test_retrieval_pipeline_strategy.py
```

前端构建：

```powershell
cd frontend
npm run build
```

## 评测与实验

项目内置 JSONL 评测集：

- `data/evaluation/retrieval_experiment_cases.jsonl`
- `data/evaluation/codedoc_agent_eval.jsonl`
- `data/evaluation/retrieval_experiment_cases_real_project.jsonl`

部分实验报告保留在：

- `outputs/experiments/retrieval_experiment_report_bge_m3_100.json`
- `outputs/experiments/agent_tool_call_experiment_report_qwen35_latest.json`
- `outputs/experiments/chunk_experiment_report_latest.json`

运行检索评测示例：

```powershell
python scripts/run_retrieval_experiments.py --prepare-mock-index
```

真实模型实验需要先准备好 chunks、向量索引、Ollama Embedding 和本地 Reranker。

## 本地生成文件说明

以下文件属于本地运行产物，不建议提交到 GitHub：

- `outputs/*_chunks.json`
- `outputs/*_vector_index.json`
- `data/*.db`
- `data/*.sqlite`
- `data/uploaded_projects/`
- `logs/`
- `.pytest_cache/`
- `.tmp_pytest_*/`
- `frontend/dist/`
- `frontend/node_modules/`

这些路径已在 `.gitignore` 中忽略。
