from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

'''
它负责管理 LangGraph Checkpoint 的配置，尤其是 SQLite 数据库文件保存在哪里。
也就是说，它回答的问题是：
Checkpoint 要写到哪个 SQLite 文件？
History 默认查多少条？
最多允许查多少条？
数据库目录不存在时要不要自动创建？
'''
class CheckpointConfig(BaseModel):
    """
    LangGraph SQLite Checkpoint 配置。

    database_path 控制 checkpoint 写入哪个 SQLite 文件；
    history_default_limit / history_max_limit 控制历史查询默认数量和最大数量。
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    database_path: str = Field(
        min_length=1,
        max_length=1000,
    )

    history_default_limit: int = Field(
        default=20,
        ge=1,
        le=100,
    )

    history_max_limit: int = Field(
        default=100,
        ge=1,
        le=500,
    )

    @classmethod
    def from_env(cls) -> "CheckpointConfig":
        """
        从环境变量读取 Checkpoint 配置，方便本地、测试和部署切换数据库文件。
        """
        return cls(
            database_path=os.getenv(
                "LANGGRAPH_CHECKPOINT_DB",
                "data/langgraph_checkpoints.sqlite",
            ),
            history_default_limit=int(
                os.getenv(
                    "LANGGRAPH_HISTORY_DEFAULT_LIMIT",
                    "20",
                )
            ),
            history_max_limit=int(
                os.getenv(
                    "LANGGRAPH_HISTORY_MAX_LIMIT",
                    "100",
                )
            ),
        )

    @property
    def resolved_database_path(self) -> str:
        """
        返回 SQLite 文件的绝对路径，避免不同启动目录造成数据库位置不一致。
        """
        return str(Path(self.database_path).resolve())

    def ensure_parent_directory(self) -> None:
        """
        确保 SQLite 数据库父目录存在。
        """
        path = Path(self.resolved_database_path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
