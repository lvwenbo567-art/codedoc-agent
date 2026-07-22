from __future__ import annotations


class ToolBusinessError(Exception):
    """
    工具内部可预期的业务异常。

    例如：
    - 项目目录不存在
    - chunks.json 不存在
    - vector_index.json 不存在
    - 检索服务返回非法数据
    """

    def __init__(
        self,
        error_code: str,
        message: str,
    ) -> None:
        super().__init__(message)

        self.error_code = error_code
        self.message = message
