from typing import Dict, List, Optional

from chunk_storage import calculate_chunk_stats, save_chunks_to_json
from chunker import build_chunks
from config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_DB_PATH
from file_loader import load_project_files
from repository import save_project_snapshot


def build_file_summaries(files: List[Dict]) -> List[Dict]:
    """
    构建文件摘要信息。
    """
    results = []

    for file in files:
        results.append(
            {
                "path": file["path"],
                "name": file["name"],
                "suffix": file["suffix"],
                "length": file["length"],
                "content_preview": file["content"][:150],
            }
        )

    return results


def build_chunk_previews(
    chunks: List[Dict],
    limit: int = 3,
) -> List[Dict]:
    """
    构建 chunk 预览信息。
    """
    results = []

    for chunk in chunks[:limit]:
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source_path": chunk["source_path"],
                "source_name": chunk["source_name"],
                "chunk_type": chunk["chunk_type"],
                "chunk_index": chunk["chunk_index"],
                "content_preview": chunk["content"][:150],
                "length": chunk["length"],
            }
        )

    return results


def scan_project(
    project_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    save_chunks: bool = False,
    output_path: str = "outputs/chunks.json",
    save_to_db: bool = True,
    db_path: str = DEFAULT_DB_PATH,
) -> Dict:
    """
    扫描项目目录，构建 chunks，并可选保存到 JSON 和 SQLite。
    """
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

    project_id: Optional[int] = None

    if save_to_db:
        project_id = save_project_snapshot(
            project_path=project_path,
            files=files,
            chunks=chunks,
            db_path=db_path,
        )

    return {
        "project_path": project_path,
        "project_id": project_id,
        "file_count": len(files),
        "chunk_count": len(chunks),
        "chunk_stats": stats,
        "files": build_file_summaries(files),
        "chunk_previews": build_chunk_previews(chunks),
        "saved_path": str(saved_path) if saved_path else None,
        "db_path": db_path if save_to_db else None,
    }