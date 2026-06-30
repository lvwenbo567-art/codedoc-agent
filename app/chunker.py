from typing import Dict, List
from document_schema import Chunk
from config import DEFAULT_CHUNK_OVERLAP,DEFAULT_CHUNK_SIZE



def validate_chunk_params(chunk_size: int, overlap: int)->None:
    """
    校验 chunk 参数是否合法。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if overlap < 0:
        raise ValueError("overlap 不能小于 0")

    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    按固定字符长度切分文本，并保留一定重叠。
    """
    validate_chunk_params(chunk_size, overlap)

    if not text or not text.strip():
        return []

    

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def get_chunk_type(suffix: str) -> str:
    """
    根据文件后缀判断 chunk 类型。
    """
    if suffix.lower() == ".py":
        return "code"

    return "document"


def build_chunks(
    files: List[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict]:
    """
    将项目文件切分为统一 chunk 结构。
    """
    results = []

    for file in files:
        content = file["content"]
        text_chunks = chunk_text(
            text=content,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        chunk_type = get_chunk_type(file["suffix"])
        
        for idx, chunk in enumerate(text_chunks):
            

            chunk_obj = Chunk(
               chunk_id=f"{file['path']}::chunk_{idx}",
               source_path=file["path"],
               source_name=file["name"],
               source_suffix=file["suffix"],
               chunk_type=chunk_type,
               chunk_index=idx,
               content=chunk,
               length=len(chunk),
           )
            results.append(chunk_obj.to_dict())
            
    return results