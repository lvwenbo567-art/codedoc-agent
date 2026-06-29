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