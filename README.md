# CodeDoc Research Agent

## 项目目标

CodeDoc Research Agent 是一个面向开源项目资料的研究助手，目标是支持读取 README、Markdown、TXT、Python 代码文件，并基于 RAG 和 Agent 能力回答项目架构、模块职责、启动方式和核心代码逻辑。

## 当前进度

Day 1：

- 完成项目目录初始化
- 实现项目文件扫描
- 支持读取 `.md`、`.txt`、`.py` 文件
- 支持输出 README 前 500 字
- 支持统计 Python 文件和 Markdown 文件

## 运行方式

```bash
python app/main.py --project_path ./test_project