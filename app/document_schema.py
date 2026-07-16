from dataclasses import dataclass, asdict
from typing import Dict,Literal,Any

@dataclass
class ProjectFile:
    """
    项目文件结构。

    表示 file_loader.py 读取出来的一个原始文件。
    """

    path: str
    name: str
    suffix: str
    content: str
    length: int

    def to_dict(self) -> Dict[str, Any]:
        """
        将项目文件对象转换为普通字典，方便 JSON 序列化和后续处理。
        """
        return asdict(self)
    
@dataclass
class Chunk:
    """
    Chunk 数据结构。

    表示 RAG 系统中的一个最小检索单元。
    """

    chunk_id: str
    source_path: str
    source_name: str
    source_suffix: str
    chunk_type: Literal["document", "code"]
    chunk_index: int
    content: str
    length: int

    def to_dict(self) -> Dict[str, Any]:
        """
        将 chunk 对象转换为普通字典，方便保存和检索。
        """
        return asdict(self)
