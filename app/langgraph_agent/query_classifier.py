from __future__ import annotations

from langgraph_agent.state import QueryType


STRUCTURE_KEYWORDS = {
    "项目结构",
    "目录结构",
    "文件结构",
    "目录",
    "模块",
    "入口文件",
    "主要文件",
    "有哪些文件",
    "project structure",
    "directory structure",
    "modules",
    "entry point",
}


DOCUMENT_KEYWORDS = {
    "readme",
    "文档",
    "说明",
    "启动",
    "运行",
    "安装",
    "部署",
    "配置方法",
    "使用方法",
    "使用说明",
    "设计文档",
    "markdown",
    ".md",
    "documentation",
    "install",
    "setup",
    "configuration",
    "deploy",
}


CODE_KEYWORDS = {
    "代码",
    "源码",
    "函数",
    "方法",
    "类",
    "定义",
    "实现",
    "调用",
    "参数",
    "返回值",
    "继承",
    "异常",
    "装饰器",
    "变量",
    "symbol",
    "function",
    "method",
    "class",
    "implementation",
    "source code",
    "defined",
    ".py",
}


class RuleBasedQueryClassifier:
    """
    Day35 使用规则分类器，保证 StateGraph 路由测试稳定。
    """

    def classify(
        self,
        query: str,
    ) -> QueryType:
        normalized_query = query.strip().lower()

        if not normalized_query:
            return "unknown"

        if self._contains_any(
            normalized_query,
            STRUCTURE_KEYWORDS,
        ):
            return "structure"

        if self._contains_any(
            normalized_query,
            DOCUMENT_KEYWORDS,
        ):
            return "document"

        if self._contains_any(
            normalized_query,
            CODE_KEYWORDS,
        ):
            return "code"

        return "unknown"

    @staticmethod
    def _contains_any(
        query: str,
        keywords: set[str],
    ) -> bool:
        return any(
            keyword.lower() in query
            for keyword in keywords
        )
