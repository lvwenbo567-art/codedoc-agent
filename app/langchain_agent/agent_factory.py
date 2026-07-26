from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import BaseTool

from langchain_agent.model_config import LangChainModelConfig
from langchain_agent.model_factory import create_chat_model
from langchain_agent.runtime_context import CodeDocRuntimeContext


CODEDOC_AGENT_SYSTEM_PROMPT = """
你是 CodeDoc Research Agent。

你的任务是根据项目中的真实代码、文档和目录结构回答用户问题。

可用工具及使用原则：

1. search_code
   用于语义搜索代码实现、类、函数和调用逻辑。

2. search_documents
   用于查询 README、启动说明、配置和设计文档。

3. get_project_structure
   用于查询目录结构、主要模块和入口文件。

4. get_symbol_definition
   当用户明确给出函数名、类名或方法名时，优先使用它获得精确 AST 定义。

5. read_file_range
   当检索结果已经提供文件路径和行号，需要查看更完整上下文时使用。

回答要求：

1. 不得编造项目中不存在的文件、函数或实现。
2. 先获取证据，再回答项目事实。
3. 精确符号问题优先使用确定性工具。
4. 搜索结果不足时，可以继续调用其他工具。
5. 最终回答应说明关键 source_path。
6. 工具失败时应根据错误信息调整调用，或明确说明无法取得证据。
7. 不允许尝试访问项目根目录之外的文件。
""".strip()


class LangChainAgentConfigurationError(ValueError):
    pass


def create_codedoc_agent(
    *,
    config: LangChainModelConfig,
    tools: list[BaseTool],
    middleware: list[Any] | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """
    创建 CodeDoc LangChain Agent。

    今天使用 LangChain v1 的高层 create_agent，不手写 LangGraph。
    """
    if config.provider == "mock":
        raise LangChainAgentConfigurationError(
            "LangChain Agent 需要支持 Tool Calling 的真实模型。"
            "mock 模式只用于普通 Chat 和结构化输出单元测试。"
        )

    model = create_chat_model(config)

    kwargs = {
        "model": model,
        "tools": tools,
        "system_prompt": CODEDOC_AGENT_SYSTEM_PROMPT,
        "context_schema": CodeDocRuntimeContext,
        "name": "codedoc_agent",
    }

    if middleware:
        kwargs["middleware"] = middleware

    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer

    try:
        return create_agent(**kwargs)
    except TypeError:
        kwargs.pop("middleware", None)
        kwargs.pop("context_schema", None)
        kwargs.pop("checkpointer", None)
        return create_agent(**kwargs)
