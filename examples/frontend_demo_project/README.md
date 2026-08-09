# Frontend Demo Project

这是一个用于测试 CodeDoc Research Agent 前端上传、扫描、切分、索引和智能问答流程的小型 Python 项目。

## 项目功能

本项目模拟一个极简文档检索系统，包含：

- 文档加载；
- 文本清洗；
- Chunk 切分；
- 关键词打分；
- Top-K 检索；
- 检索 Pipeline 编排；
- 简单 API 封装；
- pytest 测试用例。

## 启动方式

安装依赖后，可以运行测试：

```bash
python -m pytest tests
```

也可以直接调用 `src/codedoc_demo/pipeline.py` 中的 `build_search_pipeline` 函数构建检索流程。

## 核心模块

- `src/codedoc_demo/parser.py`：负责文档清洗和 Chunk 切分；
- `src/codedoc_demo/search.py`：负责关键词打分和 Top-K 检索；
- `src/codedoc_demo/pipeline.py`：负责把解析和检索流程串起来；
- `src/codedoc_demo/api.py`：提供面向外部调用的问答入口；
- `config/settings.yaml`：保存默认检索配置；
- `data/project_manifest.json`：保存项目元信息。

## 可测试问题

你可以在 CodeDoc Research Agent 前端中尝试提问：

1. `keyword_score 函数在哪里定义？它的作用是什么？`
2. `这个项目是怎么把文档变成可搜索数据的？`
3. `build_search_pipeline 函数做了什么？`
4. `README 里有没有说明怎么运行测试？`
5. `请读取 src/codedoc_demo/search.py 的第 1 行到第 60 行，并解释关键词检索逻辑。`
