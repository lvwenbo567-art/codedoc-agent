from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field


ConversationRole = Literal[
    "user",
    "assistant",
]


class ConversationTurn(BaseModel):
    """
    API 历史消息转成 LangChain Message 之前使用的内部结构。
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    role: ConversationRole
    content: str = Field(min_length=1, max_length=10000)


DEFAULT_CODEDOC_SYSTEM_PROMPT = """
你是 CodeDoc Research Agent。

你负责回答代码项目、项目文档、项目结构和 RAG 检索流程相关问题。

要求：
1. 不编造项目中不存在的文件、函数或实现。
2. 信息不足时明确说明证据不足。
3. 涉及代码标识符时保留原始名称。
4. 回答应清楚、直接，并区分事实与推断。
""".strip()


QUERY_ANALYSIS_SYSTEM_PROMPT = """
你是 CodeDoc Query Analyzer。

请判断用户问题主要属于哪一种类型：
- code：函数、类、方法、代码实现和调用关系
- document：README、启动说明、设计文档和配置说明
- structure：项目目录、模块和入口文件
- mixed：需要同时查询代码和文档
- unknown：无法可靠判断

你还需要：
1. 推荐后续工具。
2. 推荐 original、rewrite 或 multi_query 查询策略。
3. 保留函数名、类名、文件名和 API 路径。
4. classification_reason 只写简短分类依据，不输出冗长推理过程。
""".strip()



'''
history = [
    ConversationTurn(
        role="user",
        content="这个项目做什么？",
    ),
    ConversationTurn(
        role="assistant",
        content="这是一个代码研究助手。",
    ),
]
'''
def build_chat_messages(
    *,
    query: str,
    history: list[ConversationTurn] | None = None,
    system_prompt: str = DEFAULT_CODEDOC_SYSTEM_PROMPT,
    max_history_messages: int = 12,
) -> list[BaseMessage]:
    """
    把 API 输入转换为 LangChain ChatModel 需要的 Message 列表。
    """
    query = query.strip()
    system_prompt = system_prompt.strip()

    if not query:
        raise ValueError("query 不能为空")

    if not system_prompt:
        raise ValueError("system_prompt 不能为空")

    if max_history_messages < 0:
        raise ValueError("max_history_messages 不能小于 0")

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
    ]

    history = history or []

    if max_history_messages == 0:
        selected_history: list[ConversationTurn] = []
    else:
        selected_history = history[-max_history_messages:]

    for turn in selected_history:
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
        else:
            messages.append(AIMessage(content=turn.content))

    messages.append(HumanMessage(content=query))

    return messages
'''
history = [
    ConversationTurn(
        role="user",
        content="这个项目做什么？",
    ),
    ConversationTurn(
        role="assistant",
        content="这是一个代码项目研究助手。",
    ),
]

messages = build_chat_messages(
    query="它的入口文件在哪里？",
    history=history,
)

得到的消息结构大致是：

[
    SystemMessage(
        content="你是 CodeDoc Research Agent..."
    ),
    HumanMessage(
        content="这个项目做什么？"
    ),
    AIMessage(
        content="这是一个代码项目研究助手。"
    ),
    HumanMessage(
        content="它的入口文件在哪里？"
    ),
]
'''

def build_query_analysis_messages(query: str) -> list[BaseMessage]:
    """
    构建 Query 结构化分析使用的 SystemMessage 和 HumanMessage。
    """
    query = query.strip()

    if not query:
        raise ValueError("query 不能为空")

    return [
        SystemMessage(content=QUERY_ANALYSIS_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "请分析下面的项目问题：\n\n"
                f"{query}"
            )
        ),
    ]

