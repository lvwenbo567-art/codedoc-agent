# CodeDoc Research Agent Frontend

这是 CodeDoc Research Agent 的本地演示前端，用于展示一个更接近真实产品的代码仓库问答流程：

- 项目接入：输入本地项目路径，扫描文件、切分 chunks、构建向量索引；
- 项目工作台：围绕当前项目进行 Agent 问答；
- 工具与证据：展示工具调用、执行步骤、SSE 事件和原始响应；
- Human-in-the-loop：当 Agent 请求执行 `run_project_tests` 时弹出人工审核；
- 评测反馈：导入 eval report、提交用户反馈、查看 bad cases。

## 启动后端

在项目根目录 `codedoc-agent` 下启动 FastAPI：

```powershell
$env:LANGCHAIN_CHAT_PROVIDER="openai_compatible"
$env:LANGCHAIN_CHAT_MODEL="qwen3.5:4b"
$env:LANGCHAIN_CHAT_BASE_URL="http://localhost:11434/v1"
$env:LANGCHAIN_CHAT_API_KEY="EMPTY"
$env:LANGCHAIN_CHAT_TIMEOUT_SECONDS="120"
$env:LANGCHAIN_CHAT_MAX_TOKENS="500"
$env:LANGCHAIN_CHAT_MAX_RETRIES="0"

uvicorn api_main:app --reload --app-dir app
```

## 安装依赖并启动前端

```powershell
cd frontend
npm install
npm run dev
```

访问：

```text
http://127.0.0.1:5173
```

Vite 会把 `/langgraph/*`、`/agent-quality/*`、`/version` 自动代理到 `http://127.0.0.1:8000`。

## 页面流程

1. 进入“项目接入”页。
2. 选择一种项目接入方式：

方式 A：上传 zip 项目包。

前端会调用：

```text
POST /project-upload/zip
```

后端会解压到：

```text
data/uploaded_projects/
```

方式 B：输入后端本地项目路径，例如：

```text
test_project
```

3. 点击“扫描并生成 chunks”，调用：

```text
POST /scan
```

4. 点击“构建向量索引”，调用：

```text
POST /index
```

5. 进入“项目工作台”开始提问。

如果已经有 `outputs/test_project_chunks.json` 和
`outputs/test_project_vector_index_bge_m3.json`，也可以点击“使用 test_project 示例”直接进入工作台。

## 推荐测试问题

普通工具调用：

```text
keyword_score 函数在哪里定义？请读取它附近源码并解释作用。
```

人工审核：

```text
请运行 tests/test_project_test_tools.py，验证 run_project_tests 工具是否正常。
```

项目结构：

```text
这个项目有哪些主要目录和模块？请先查看项目结构再回答。
```
