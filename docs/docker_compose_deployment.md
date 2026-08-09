# Docker Compose 部署说明

本文档说明如何使用 Docker Compose 启动 CodeDoc Research Agent 的本地演示环境。

## 1. 服务拆分

```text
frontend  -> React/Vite 前端工作台
api       -> FastAPI 后端
qdrant    -> 向量数据库
redis     -> 缓存/会话扩展位
ollama    -> 默认使用宿主机 Ollama；容器 Ollama 可选
sqlite    -> 以 volume 形式持久化在 api 容器的 /app/data 下
```

当前项目主线使用 SQLite 保存 checkpoint、长期记忆、ingestion job 和反馈数据。Redis 在 Compose 中作为后续缓存、分布式任务或会话加速的预留服务，不是当前核心链路的强依赖。

## 2. 为什么 SQLite 不单独起服务

SQLite 不是一个独立服务，它是一个文件数据库。Compose 中通过 volume 挂载：

```text
codedoc_data:/app/data
```

因此下面这些文件会被持久化：

- `/app/data/langgraph_checkpoints.sqlite`
- `/app/data/codedoc_memory.sqlite`
- `/app/data/ingestion_jobs.sqlite`
- `/app/data/codedoc.db`

## 3. 默认方式：使用宿主机 Ollama

默认 Compose 配置使用宿主机已经安装好的 Ollama，这样不用拉取 2GB+ 的 `ollama/ollama` Docker 镜像。

先在宿主机确认模型服务可用：

```powershell
ollama list
Invoke-RestMethod http://localhost:11434/api/tags
```

至少需要：

- `qwen3.5:4b`
- `bge-m3`

API 容器会通过下面的 Docker Desktop 特殊地址访问宿主机 Ollama：

```text
http://host.docker.internal:11434
```

## 4. 可选方式：使用容器 Ollama

如果你希望模型服务也完全运行在 Docker 中，可以显式启用 `container-models` profile：

```powershell
docker compose --profile container-models up -d ollama
docker compose --profile container-models up ollama-init
```

注意：首次拉取 `ollama/ollama` 镜像和模型会比较慢。

## 5. 启动完整服务

```powershell
docker compose up --build api frontend qdrant redis
```

访问：

```text
Frontend: http://127.0.0.1:5173
FastAPI:  http://127.0.0.1:8000/docs
Qdrant:   http://127.0.0.1:6333/dashboard
Ollama:   http://127.0.0.1:11434
```

## 6. 健康检查

查看服务状态：

```powershell
docker compose ps
```

单独检查后端：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

单独检查 Qdrant：

```powershell
Invoke-RestMethod http://127.0.0.1:6333/collections
```

单独检查宿主机 Ollama：

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

单独检查 Redis：

```powershell
docker compose exec redis redis-cli ping
```

预期返回：

```text
PONG
```

## 7. 前端使用流程

1. 打开 `http://127.0.0.1:5173`。
2. 使用示例项目或上传 ZIP 项目。
3. 点击“扫描”生成 chunks。
4. 点击“构建索引”写入 Qdrant。
5. 进入“问答”提问。
6. 遇到测试运行等工具调用时，在前端执行 approve / reject / edit。

## 8. 常见问题

### 8.1 API 一直不 healthy

先看日志：

```powershell
docker compose logs -f api
```

常见原因：

- 宿主机 Ollama 没启动或模型不存在。
- Qdrant healthcheck 未通过。
- 本地端口被占用。

### 8.2 Ollama 模型不存在

宿主机检查：

```powershell
ollama list
```

如果没有模型，重新执行：

```powershell
ollama pull qwen3.5:4b
ollama pull bge-m3
```

### 8.3 为什么默认不启动容器 Ollama

`ollama/ollama` 镜像和模型文件都比较大，国内网络下首次拉取可能非常慢。你的开发机通常已经安装了 Ollama，因此默认使用宿主机 Ollama 更适合本地开发和 GitHub 演示。

### 8.4 容器里默认为什么用 mock rerank

当前本地真实 reranker 依赖 Windows 本机路径，例如：

```text
D:/models/bge-reranker-v2-m3
```

这个路径在 Docker 容器内不可复现，因此 Compose 默认使用 `mock` rerank，保证 GitHub 用户可以启动完整服务。真实 reranker 可以后续通过挂载模型目录和安装 `sentence-transformers` 扩展。

## 9. 停止与清理

停止服务：

```powershell
docker compose down
```

如果要连数据和模型一起清理：

```powershell
docker compose down -v
```

注意：`-v` 会删除 Qdrant、Ollama、SQLite 等 volume 数据。
