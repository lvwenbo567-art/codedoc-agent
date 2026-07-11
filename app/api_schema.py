from pydantic import BaseModel

from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DB_PATH,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_INDEX_PATH,
)


class ScanRequest(BaseModel):
    """
    /scan 接口请求体。
    """

    project_path: str
    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_CHUNK_OVERLAP
    save_chunks: bool = False
    output_path: str = "outputs/chunks.json"
    save_to_db: bool = True
    db_path: str = DEFAULT_DB_PATH

class SearchRequest(BaseModel):
    """
    /search 接口请求体。
    """

    chunks_path: str = "outputs/chunks.json"
    query: str
    top_k: int = 5

class EvalRequest(BaseModel):
    """
    /eval 接口请求体。
    """

    chunks_path: str = "outputs/chunks.json"
    eval_path: str = "data/eval_queries.json"
    top_k: int = 5

class IndexRequest(BaseModel):
    """
    /index 接口请求体。
    """

    chunks_path: str = "outputs/chunks.json"
    output_path: str = DEFAULT_VECTOR_INDEX_PATH
    model_name: str = DEFAULT_EMBEDDING_MODEL
    dimension: int = DEFAULT_EMBEDDING_DIMENSION