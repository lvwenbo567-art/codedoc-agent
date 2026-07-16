"""
测试项目配置模块。

这个文件用于测试 CodeDoc 是否能识别配置常量、环境变量说明和默认路径。
"""

DEFAULT_DATABASE_PATH = "data/demo.db"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_TOP_K = 3


def get_database_url(database_path: str = DEFAULT_DATABASE_PATH) -> str:
    """
    根据数据库文件路径生成 SQLite 连接 URL。
    """
    return f"sqlite:///{database_path}"


def get_runtime_config() -> dict:
    """
    返回测试项目运行时配置，供启动流程和 API 模块使用。
    """
    return {
        "database_path": DEFAULT_DATABASE_PATH,
        "log_level": DEFAULT_LOG_LEVEL,
        "top_k": DEFAULT_TOP_K,
    }
