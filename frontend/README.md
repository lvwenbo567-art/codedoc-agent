# CodeDoc Frontend

React + Vite 前端工作台，用于演示 CodeDoc Research Agent 的项目接入、代码仓库问答、人工审核和评测反馈流程。

## 功能

- 上传 ZIP 项目或使用示例项目。
- 扫描项目并构建 chunks。
- 构建向量索引。
- 多项目、多会话切换。
- Agent 问答与工具调用结果展示。
- HITL 人工审核 approve / reject / edit。
- 导入评测报告，提交用户反馈并沉淀 Bad Case。

## 启动

先启动后端：

```powershell
cd codedoc-agent

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

再启动前端：

```powershell
cd frontend
npm install
npm run dev
```

访问：

```text
http://127.0.0.1:5173
```

## 构建检查

```powershell
npm run build
```

构建产物位于 `frontend/dist/`，该目录属于本地生成文件，不提交到 GitHub。
