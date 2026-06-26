## 当前进度​### Day 1​- 完成项目目录初始化- 实现项目文件扫描- 支持读取 `.md`、`.txt`、`.py` 文件- 支持输出 README 前 500 字​### Day 2​- 新增 `code_parser.py`- 使用 Python AST 解析 `.py` 文件- 支持提取函数名、类名、起始行号和 docstring- 在 `main.py` 中输出 Python 代码结构
### Day 3

- 新增 `llm_client.py`
- 设计 `LLMClient` 类，作为统一的大模型调用入口
- 新增 `config.py` 基础配置，包括模型名、base_url 和 api_key
- 支持对 README 内容生成模拟摘要
- 为后续接入 OpenAI-compatible API 和 vLLM 推理服务做准备