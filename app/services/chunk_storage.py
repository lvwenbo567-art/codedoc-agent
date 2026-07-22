import json
from pathlib import Path
from typing import Dict,List

def save_chunks_to_json(
        chunks:List[Dict],
        output_path:str="outputs/chunks.json",
)->Path:
    """
    将 chunks 保存为 JSON 文件。
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(
         json.dumps(chunks,ensure_ascii=False,indent=2),
         encoding="utf-8",
    )

    return path

def load_chunks_from_json(input_path:str)->List[Dict]:
    """
    从 JSON 文件中读取 chunks。
    """
    path=Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"chunks 文件不存在： {input_path}")
    
    content=path.read_text(encoding="utf-8")
    return json.loads(content)

def calculate_chunk_stats(chunks:List[Dict])->Dict:
    """
    统计 chunk 数量、代码 chunk 数量、文档 chunk 数量和平均长度。
    """
    total=len(chunks)
    code_count=len([c for c in chunks if c["chunk_type"]=="code"])
    document_count=len([c for c in chunks if c["chunk_type"] == "document"])

    if total==0:
        avg_length=0
    else:
        avg_length=sum(c["length"] for c in chunks) / total

    return {
        "total": total,
        "code_count": code_count,
        "document_count": document_count,
        "avg_length": avg_length,
    }