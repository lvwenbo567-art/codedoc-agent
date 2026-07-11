## 当前进度​### Day 1​- 完成项目目录初始化- 实现项目文件扫描- 支持读取 `.md`、`.txt`、`.py` 文件- 支持输出 README 前 500 字​### Day 2​- 新增 `code_parser.py`- 使用 Python AST 解析 `.py` 文件- 支持提取函数名、类名、起始行号和 docstring- 在 `main.py` 中输出 Python 代码结构
### Day 3

- 新增 `llm_client.py`
- 设计 `LLMClient` 类，作为统一的大模型调用入口
- 新增 `config.py` 基础配置，包括模型名、base_url 和 api_key
- 支持对 README 内容生成模拟摘要
- 为后续接入 OpenAI-compatible API 和 vLLM 推理服务做准备

### Day 3 加做

- 新增 `chunker.py`
- 实现固定长度文本切分
- 支持 chunk overlap，减少语义断裂
- 将项目文件转换为统一 chunk 结构
- 在 `main.py` 中输出文档 chunk、代码 chunk 和总 chunk 数量


### Day 5

- 对前几天的基础模块进行收尾整理
- 确认 `file_loader.py`、`code_parser.py`、`llm_client.py`、`chunker.py` 可以串联运行
- 为 `chunker.py` 增加基础测试
- 为 `code_parser.py` 增加基础测试
- 初步理解 pytest 的作用：用自动化方式验证函数行为是否符合预期
- 当前项目已经完成从项目文件读取到 chunk 构建的最小链路

### Day 6

- 新增 `document_schema.py`
- 使用 `dataclass` 定义 `ProjectFile` 和 `Chunk` 数据结构
- 对 chunk 数据结构进行标准化，为后续 Embedding、向量库和数据库存储做准备
- 新增 `logger.py`
- 使用 logging 替代部分 print，支持控制台日志和文件日志
- 为 `chunker.py` 增加字段完整性测试
- 当前项目进入基础工程化整理阶段

### Day 7

- 整理 `config.py`，集中管理支持文件后缀、chunk_size、overlap 和 mock LLM 配置
- 修改 `file_loader.py`，使用 `ProjectFile` dataclass 统一文件结构
- 新增 `tests/test_file_loader.py`
- 测试支持文件扫描、文件 schema、路径不存在异常、非目录异常
- 修改 `main.py`，支持通过命令行传入 `--chunk_size` 和 `--overlap`
- 当前项目的数据入口层更加稳定，为后续 Embedding 和 RAG 检索做准备

### Day 8

- 对 `chunker.py` 进行工程化整理
- 新增 `validate_chunk_params()`，统一校验 `chunk_size` 和 `overlap`
- 新增 `get_chunk_type()`，根据文件后缀区分 `document` 和 `code`
- 使用 `Chunk` dataclass 统一 chunk 输出结构
- 补强 `tests/test_chunker.py`
- 测试正常切分、空文本、非法参数、chunk 字段完整性和 chunk 类型判断
- 修改 `main.py`，输出前 3 个 chunk 示例
- 当前完成了 RAG 数据准备链路中的 `file -> chunk` 阶段

### Day 9

- 修复 `document_schema.py` 中 `Any` 未导入的问题
- 完成 `file_loader -> chunker -> main.py` 链路联调
- 新增 `chunk_storage.py`
- 支持将 chunks 保存为 JSON 文件
- 支持从 JSON 文件读取 chunks
- 新增 chunk 统计功能，包括总数量、代码 chunk 数量、文档 chunk 数量和平均长度
- 新增 `tests/test_chunk_storage.py`
- 当前 RAG 数据准备层已经支持：文件读取、chunk 构建、chunk 统计和 JSON 持久化

### Day 10

- 新增 `retriever.py`
- 实现关键词检索版 Retriever v0
- 新增 `extract_query_terms()`，用于从 query 中提取关键词
- 新增 `score_chunk()`，根据 query 与 chunk 内容、文件名匹配情况计算相关性分数
- 新增 `search_chunks()`，支持从 chunks 中返回 Top-K 相关片段
- 新增 `tests/test_retriever.py`
- 测试 query 解析、chunk 打分、Top-K 检索、无命中和非法 top_k
- 修改 `main.py`，支持 `--query` 和 `--top_k`
- 当前项目已经初步具备 RAG 检索层雏形：file -> chunk -> retrieve

### Day 11

- 新增 `search_service.py`
- 实现 `build_search_results()`，将检索到的 chunks 转换为统一搜索结果结构
- 实现 `search_chunks_from_json()`，支持直接从 `chunks.json` 中读取 chunks 并执行检索
- 新增 `search_cli.py`
- 支持通过命令行直接检索已有 chunks 文件
- 新增 `tests/test_search_service.py`
- 测试搜索结果结构化、从 JSON 检索和无命中情况
- 当前项目已经初步拆分出离线数据准备阶段和在线检索阶段

### Day 12

- 新增 `retrieval_eval.py`
- 实现 `calculate_hit_rate()`，用于判断 Top-K 检索结果是否至少命中一个正确 chunk
- 实现 `calculate_recall()`，用于计算正确 chunks 的召回比例
- 实现 `calculate_mrr()`，用于衡量第一个正确结果的排名
- 实现 `evaluate_single_query()` 和 `evaluate_queries()`，支持单条和多条 query 的检索评估
- 新增 `eval_cli.py`
- 支持基于 `chunks.json` 和人工标注的 `eval_queries.json` 运行检索评估
- 新增 `tests/test_retrieval_eval.py`
- 当前项目已经具备关键词检索和基础检索评估能力

### Day 13

- 新增 `api_main.py`
- 引入 FastAPI，创建 CodeDoc Research Agent API 后端入口
- 新增 `/health` 接口，用于服务健康检查
- 新增 `/version` 接口，用于查看当前 API 版本和阶段
- 新增 `/config` 接口，用于查看项目基础配置
- 新增 `tests/test_api_main.py`
- 使用 FastAPI `TestClient` 测试基础接口
- 当前项目开始从命令行工具升级为可通过 HTTP 调用的后端服务

### Day 14

- 新增 `project_service.py`
- 将项目扫描、chunk 构建、chunk 统计逻辑从 `main.py` 中抽取到 service 层
- 新增 `api_schema.py`
- 使用 Pydantic 定义 `/scan` 接口请求体 `ScanRequest`
- 修改 `api_main.py`
- 新增 `POST /scan` 接口，支持通过 HTTP 请求扫描项目并构建 chunks
- `/scan` 支持 `project_path`、`chunk_size`、`overlap`、`save_chunks` 和 `output_path`
- 新增 `tests/test_api_scan.py`
- 测试 `/scan` 成功扫描、路径不存在、非法 overlap 和保存 chunks 的情况
- 当前项目已经从基础 FastAPI 入口推进到业务 API 接口

### Day 15

- 修改 `api_schema.py`
- 新增 `SearchRequest`，用于定义 `/search` 接口请求体
- 修改 `api_main.py`
- 新增 `POST /search` 接口
- `/search` 支持从 `chunks.json` 中根据 query 检索 Top-K chunks
- 复用 `search_service.py` 中的 `search_chunks_from_json()`
- 新增 `tests/test_api_search.py`
- 测试 `/search` 成功检索、无命中、chunks 文件不存在、空 query、非法 top_k 等情况
- 当前项目已经支持通过 HTTP API 完成项目扫描和 chunks 检索

### Day 16

- 新增 `api_response.py`
- 实现统一成功响应 `success_response()`
- 实现统一失败响应 `error_response()`
- 实现 HTTP 状态码到错误码的转换
- 修改 `api_schema.py`
- 新增 `EvalRequest`，用于定义 `/eval` 接口请求体
- 新增 `eval_service.py`
- 将评估文件读取和检索评估流程封装到 service 层
- 修改 `api_main.py`
- 新增 `POST /eval` 接口，支持通过 HTTP API 执行检索评估
- 为 `HTTPException` 和请求参数校验错误增加统一异常处理
- 修改 `/health`、`/version`、`/config`、`/scan`、`/search`，统一返回 `{success, data}` 格式
- 新增 `tests/test_api_response.py`
- 新增 `tests/test_api_eval.py`
- 当前项目已经支持通过 API 完成项目扫描、chunks 检索和检索评估

### Day 17

- 修改 `config.py`
- 新增默认数据库路径 `DEFAULT_DB_PATH`
- 新增 `db.py`
- 支持初始化 SQLite 数据库
- 新增 `projects`、`files`、`chunks` 三张表
- 新增 `repository.py`
- 支持保存一次项目扫描快照
- 支持查询项目扫描记录
- 支持查询 chunks 列表
- 支持根据 `chunk_id` 查询 chunk 详情
- 修改 `project_service.py`
- `/scan` 支持将扫描结果写入 SQLite
- 修改 `api_schema.py`
- `ScanRequest` 新增 `save_to_db` 和 `db_path`
- 修改 `api_main.py`
- 新增 `GET /projects`
- 新增 `GET /chunks`
- 新增 `GET /chunks/{chunk_id}`
- 当前项目已经支持将 RAG 数据从 JSON 文件升级为 SQLite 元数据存储


### Day 18

- 修改 `db.py`
- 启用 SQLite 外键约束
- 为项目、文件和 chunk 常用查询字段增加索引
- 修改 `repository.py`
- 新增统一分页参数校验
- 项目、文件和 chunk 查询支持 `limit` 和 `offset`
- 新增 `get_project_by_id()`
- 新增 `list_files()` 和 `get_file_by_id()`
- 新增 `get_chunk_by_id()`
- 为项目扫描数据写入增加事务回滚
- 修改 `api_main.py`
- 新增 `GET /projects/{project_id}`
- 新增 `GET /files`
- 新增 `GET /files/{file_id}`
- 修改 `GET /chunks`，支持分页
- 修改 chunk 详情接口，改为通过数据库自增 ID 查询
- 新增数据库查询和分页测试
- 当前 FastAPI + SQLite 后端基础阶段基本完成

### Day 19

- 修改 `config.py`
- 新增 Embedding 模型、向量维度和向量索引路径配置
- 新增 `embedding_client.py`
- 实现本地确定性哈希 Embedding v0
- 支持单条文本和批量文本向量化
- 对生成向量进行 L2 归一化
- 新增 `vector_store.py`
- 实现余弦相似度计算
- 支持向量索引的 JSON 保存和读取
- 新增 `index_service.py`
- 支持从 `chunks.json` 为所有 chunks 生成向量
- 向量记录同时保存 chunk metadata 和 embedding
- 修改 `api_schema.py`
- 新增 `IndexRequest`
- 修改 `api_main.py`
- 新增 `POST /index`
- 新增 Embedding、向量存储、索引服务和 API 测试
- 当前项目正式进入手写向量 RAG 阶段