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

### Day 20

- 新增 `vector_search_service.py`
- 支持将用户 query 转换成 Embedding
- 支持计算 query 与 chunk 向量的余弦相似度
- 支持按照相似度从高到低返回 Top-K chunks
- 支持通过 `chunk_type` 过滤代码或文档 chunks
- 向量检索结果返回 rank、score 和 chunk metadata
- 新增 `vector_search_cli.py`
- 支持通过命令行执行向量检索
- 修改 `api_schema.py`
- 新增 `VectorSearchRequest`
- 修改 `api_main.py`
- 新增 `POST /vector_search`
- 新增向量检索 service 和 API 测试
- 当前项目已经完成手写向量索引和向量检索基础链路

### Day 21

- 修改 `vector_search_service.py`
- 支持在内部检索时返回完整 chunk 内容
- 新增 `prompt_builder.py`
- 支持将 Top-K 检索结果构建为带来源编号的 RAG Prompt
- 支持限制上下文最大长度
- 新增 `citation_builder.py`
- 支持返回结构化引用来源
- 扩展 `llm_client.py`
- 新增 Mock Chat 模型调用
- 新增 `rag_service.py`
- 实现检索、Prompt、生成和引用的基础 RAG 闭环
- 修改 `api_schema.py`
- 新增 `AskRequest`
- 修改 `api_main.py`
- 新增 `POST /ask`
- 新增 Prompt、引用、RAG Service 和 API 测试
- 当前项目已经具备基础向量 RAG 问答能力

### Day 22

- 重构 `llm_client.py`
- 新增 `ChatConfig` dataclass
- 新增 `ChatClient`
- 支持 `mock` 和 `openai_compatible` 两种 Chat Provider
- 支持通过环境变量配置模型、Base URL 和 API Key
- 支持调用 OpenAI-compatible `/chat/completions`
- 增加模型请求超时、连接错误和返回结构校验
- 修改 `prompt_builder.py`
- 将 RAG Prompt 拆分为 System Message 和 User Message
- 新增 `answer_quality.py`
- 支持检查回答中的引用是否合法
- 修改 `rag_service.py`
- `/ask` 支持切换真实 Chat 模型
- 修改 `api_schema.py`
- 增加 Chat Provider、温度和最大输出长度参数
- 修改 `api_response.py`
- 增加模型服务 502 和 504 错误码
- 新增 `.env.example`
- 新增 Chat Client 和回答质量测试

### Day 23

- 重构 `embedding_client.py`
- 新增 `EmbeddingConfig`
- 支持 `mock`、`ollama` 和 `openai_compatible` 三种 Embedding Provider
- 支持 Ollama `/api/embed`
- 支持 OpenAI-compatible `/embeddings`
- 向量索引新增 metadata 和格式版本
- 记录 Embedding Provider、模型、维度、归一化方式和构建时间
- 查询前校验索引与查询使用的 Provider、模型和维度是否一致
- 新增 `embedding_probe.py`
- 新增真实 Embedding 客户端和索引元数据测试

### Day 24

- 新增 `batch_utils.py`
- 支持将 chunks 按 `batch_size` 分批向量化
- `EmbeddingClient` 增加有限重试
- 支持超时、连接失败、HTTP 429 和 5xx 的有限重试
- 使用指数退避控制重试间隔
- 记录 Embedding 请求次数和重试次数
- 向量索引构建返回批次数、耗时和请求统计
- 索引 metadata 保存 `build_stats`
- 向量索引通过临时文件原子性替换
- `/index` 支持配置 `batch_size`
- 新增批处理、重试、建库统计和原子保存测试
- 新增 Mock 与真实 Embedding 检索对比记录模板

### Day 25

- 新增 `content_hash.py`
- 使用 SHA-256 计算 chunk 内容哈希
- 向量记录增加 `content_hash`
- 新增 `index_update_service.py`
- 支持增量构建向量索引
- 未变化 chunk 直接复用旧 embedding
- 新增和修改 chunk 重新向量化
- 删除的 chunk 不再写入新索引
- 相同内容只调用一次 Embedding
- 保留重复内容对应的独立 metadata
- 增加新增、修改、删除、复用和去重统计
- 修改 `batch_utils.py`
- 新增基于 `yield` 的 `iter_batches()`
- `/index` 新增 `incremental` 参数
- 增加内容哈希、生成器批次和增量索引测试

### Day 26

- 新增 `hybrid_search_service.py`
- 支持关键词检索和向量检索的混合召回
- 支持对两个检索通道的分数做 Min-Max 归一化
- 支持通过 `keyword_weight` 和 `vector_weight` 调整融合权重
- 支持按 `chunk_id` 去重，同一个 chunk 可同时标记为 keyword/vector 命中
- 支持返回 `keyword_score`、`vector_score`、`final_score` 和 `matched_by`
- 修改 `api_schema.py`
- 新增 `HybridSearchRequest`
- 修改 `api_main.py`
- 新增 `POST /hybrid_search`
- `/version` 阶段更新为 `day26-hybrid-search`
- 新增混合检索 service 和 API 测试
- 新增 `python_practice/processor.py`
- 新增 `python_practice/file_utils.py`
- 新增 `python_practice/chunk_utils.py`
- 补充 Day26 Python 专项 pytest，用于练习文本处理、JSON 读写、环境变量覆盖、chunk 过滤和生成器分批

### Day 27

- 新增 `rerank_client.py`
- 支持 Mock Reranker，用于单元测试和流程打通
- 支持 Sentence Transformers CrossEncoder Provider
- 支持通过本地路径加载离线 Rerank 模型
- 新增 `rerank_service.py`
- 支持对 Hybrid Search 候选结果进行精排
- 保留召回阶段排名 `retrieval_rank` 和精排分数 `rerank_score`
- 新增 `retrieval_pipeline.py`
- 实现 Hybrid Search Top-K → Rerank Top-K
- 新增 `rerank_probe.py`
- 支持命令行测试真实 CrossEncoder 模型
- 修改 `api_schema.py`
- 新增 `RerankSearchRequest`
- 修改 `api_main.py`
- 新增 `POST /rerank_search`
- `/version` 阶段更新为 `day27-rerank`
- `/ask` 支持 `vector`、`hybrid` 和 `rerank` 三种检索模式
- 新增 Rerank Client、Service、Pipeline 和 API 测试
- 新增 `docs/day27_rerank_comparison.md`
- 新增 SQL 专项目录 `sql_practice`
- 完成 12 条以上 SQL 查询、参数化 SQLite 查询和专项测试

### Day 28

- 修改 `rerank_client.py`
- 新增 `RerankServiceError`
- 真实 CrossEncoder 加载失败和推理失败会包装为明确的 Rerank 服务异常
- 修改 `retrieval_pipeline.py`
- Rerank 服务异常时自动降级为 Hybrid Search
- 新增 `rerank_applied`、`degraded`、`degrade_reason` 和 `rerank_duration_ms`
- 只捕获 `RerankServiceError`，不会吞掉参数错误和数据结构错误
- 新增 `retrieval_metrics.py`
- 支持计算 Hit@K 和 MRR
- 新增 `rerank_eval_service.py`
- 支持对比 Hybrid Search 与 Hybrid + Rerank 的检索指标
- 新增 `data/rerank_eval_queries.json`
- 建立 10 条本地小型 Rerank 评测集
- 修改 `api_schema.py` 和 `api_main.py`
- 新增 `POST /rerank_eval`
- `/version` 阶段更新为 `day28-rerank-stability-eval`
- 新增 Rerank 降级、检索指标、评估服务和 API 测试
- 新增 `docs/day28_rerank_evaluation.md`
- 新增 SQL 专项 `day28_group_join.sql`
- 新增 `day28_report_repository.py`
- 支持 GROUP BY、HAVING、INNER JOIN、LEFT JOIN 和聚合统计测试
