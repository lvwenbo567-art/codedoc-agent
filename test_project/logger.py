"""
测试项目日志模块。

这个文件用于测试日志初始化函数和日志级别配置。
"""

import logging

from config import DEFAULT_LOG_LEVEL


def setup_logger(name: str = "test_project") -> logging.Logger:
    """
    初始化并返回测试项目 logger。
    """
    logger = logging.getLogger(name)
    logger.setLevel(DEFAULT_LOG_LEVEL)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
