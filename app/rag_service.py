from typing import Dict, Optional

from citation_builder import build_citations
from config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_RAG_TOP_K,
    DEFAULT_VECTOR_INDEX_PATH,
)
from llm_client import generate_chat_response
from prompt_builder import build_rag_prompt
from vector_search_service import search_vector_index_from_file


def ask_from_vector_index(
    query: str,
    index_path: str = DEFAULT_VECTOR_INDEX_PATH,
    top_k: int = DEFAULT_RAG_TOP_K,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    chat_model: str = DEFAULT_CHAT_MODEL,
    chunk_type: Optional[str] = None,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> Dict:
    """
    执行基础 RAG 问答流程。
    """
    retrieval_result = search_vector_index_from_file(
        query=query,
        index_path=index_path,
        top_k=top_k,
        model_name=embedding_model,
        dimension=dimension,
        chunk_type=chunk_type,
        include_content=True,
    )

    retrieved_chunks = retrieval_result["results"]

    prompt = build_rag_prompt(
        query=query,
        retrieved_chunks=retrieved_chunks,
        max_context_chars=max_context_chars,
    )

    answer = generate_chat_response(
        prompt=prompt,
        model_name=chat_model,
    )

    citations = build_citations(retrieved_chunks)

    return {
        "query": query,
        "answer": answer,
        "chat_model": chat_model,
        "embedding_model": embedding_model,
        "top_k": top_k,
        "chunk_type": chunk_type,
        "retrieval_count": len(retrieved_chunks),
        "citations": citations,
    }