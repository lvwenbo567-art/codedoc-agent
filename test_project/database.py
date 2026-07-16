"""
测试项目数据库模块。

这个文件模拟一个很小的 SQLite 数据访问层，用于测试数据库相关问题检索。
"""

from config import DEFAULT_DATABASE_PATH


def init_database(database_path: str = DEFAULT_DATABASE_PATH) -> dict:
    """
    初始化数据库，并返回创建的表信息。
    """
    tables = [
        "projects",
        "documents",
        "chunks",
    ]

    return {
        "database_path": database_path,
        "tables": tables,
    }


def save_document(document_id: str, content: str) -> dict:
    """
    保存文档内容，返回一条模拟的文档记录。
    """
    return {
        "document_id": document_id,
        "content": content,
        "length": len(content),
    }


def list_chunks(document_id: str) -> list[dict]:
    """
    查询指定文档下的 chunks。
    """
    return [
        {
            "chunk_id": f"{document_id}::chunk_0",
            "document_id": document_id,
            "content": "这是一个示例 chunk。",
        }
    ]
