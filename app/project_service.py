from typing import Dict, List

from chunk_storage import calculate_chunk_stats, save_chunks_to_json
from chunker import build_chunks
from config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from file_loader import load_project_files
from logger import setup_logger


logger = setup_logger()


def build_file_summaries(files: List[Dict]) -> List[Dict]:
    """
    构建文件摘要信息。

    不返回完整 content，避免 API 响应过大。
    """
    summaries = []

    for file in files:
        summaries.append(
            {
                "path": file["path"],
                "name": file["name"],
                "suffix": file["suffix"],
                "length": file["length"],
            }
        )

    return summaries


def build_chunk_previews(
    chunks: List[Dict],
    limit: int = 3,
) -> List[Dict]:
    """
    构建 chunk 预览信息。
    """
    previews = []

    for chunk in chunks[:limit]:
        previews.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source_name": chunk["source_name"],
                "source_path": chunk["source_path"],
                "chunk_type": chunk["chunk_type"],
                "chunk_index": chunk["chunk_index"],
                "length": chunk["length"],
                "content_preview": chunk["content"][:150],
            }
        )

    return previews


def scan_project(
    project_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    save_chunks: bool = False,
    output_path: str = "outputs/chunks.json",
) -> Dict:
    """
    扫描项目目录，构建 chunks，并返回结构化结果。
    """
    logger.info(
        "开始扫描项目，project_path=%s, chunk_size=%s, overlap=%s",
        project_path,
        chunk_size,
        overlap,
    )

    files = load_project_files(project_path)

    chunks = build_chunks(
        files=files,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    stats = calculate_chunk_stats(chunks)

    saved_path = None

    if save_chunks:
        saved_path = save_chunks_to_json(
            chunks=chunks,
            output_path=output_path,
        )

    result = {
        "project_path": project_path,
        "file_count": len(files),
        "chunk_count": len(chunks),
        "chunk_stats": stats,
        "files": build_file_summaries(files),
        "chunk_previews": build_chunk_previews(chunks),
        "saved_path": str(saved_path) if saved_path else None,
    }

    logger.info(
        "项目扫描完成，file_count=%s, chunk_count=%s",
        result["file_count"],
        result["chunk_count"],
    )

    return result