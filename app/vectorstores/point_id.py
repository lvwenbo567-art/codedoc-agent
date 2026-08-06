from __future__ import annotations

import uuid

CODEDOC_POINT_NAMESPACE = uuid.UUID("d8cd6246-2c98-4dbf-83df-d118cae7df41")
#这是一个固定的“命名空间”，你可以先把它当成项目内部固定的盐值/规则标识。

def build_vector_point_id(*, project_id: int, chunk_id: str) -> str:
    """根据 project_id + chunk_id 生成稳定的 Qdrant Point UUID。"""
    if project_id <= 0:
        raise ValueError("project_id 必须大于 0")

    normalized_chunk_id = str(chunk_id).strip()

    if not normalized_chunk_id:
        raise ValueError("chunk_id 不能为空")

    identity = f"project:{project_id}:chunk:{normalized_chunk_id}"

    return str(uuid.uuid5(CODEDOC_POINT_NAMESPACE, identity))#相同输入，永远生成相同输出。