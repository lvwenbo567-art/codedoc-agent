"""
测试项目 API 模块。

这个文件模拟后端接口函数，方便测试“项目有哪些接口”和“如何查询 chunks”。
"""

from database import init_database, list_chunks, save_document


def health_check() -> dict:
    """
    返回服务健康状态。
    """
    return {
        "status": "ok",
        "service": "test-project",
    }
def get_project_info() -> dict:
    """
    返回测试项目的基础信息。
    """
    return {
        "name": "test-project",
        "version": "0.1.0",
    }

def create_document_api(document_id: str, content: str) -> dict:
    """
    模拟创建文档接口，内部调用 save_document。
    """
    return save_document(
        document_id=document_id,
        content=content,
    )


def list_chunks_api(document_id: str) -> dict:
    """
    模拟查询 chunks 接口，返回指定文档的 chunk 列表。
    """
    return {
        "document_id": document_id,
        "chunks": list_chunks(document_id),
    }


def startup() -> dict:
    """
    启动 API 服务前初始化数据库。
    """
    return init_database()
