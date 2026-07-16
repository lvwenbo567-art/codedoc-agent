from typing import Dict, Optional

from answer_quality import evaluate_answer_quality
from citation_builder import build_citations
from config import (
    DEFAULT_CHAT_API_KEY,
    DEFAULT_CHAT_BASE_URL,
    DEFAULT_CHAT_MAX_TOKENS,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHAT_PROVIDER,
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_CHAT_TIMEOUT_SECONDS,
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_RAG_TOP_K,
    DEFAULT_VECTOR_INDEX_PATH,
)
from llm_client import generate_chat_response
from prompt_builder import build_rag_messages
from vector_search_service import (
    search_vector_index_from_file,
)


def ask_from_vector_index(
    query: str,
    index_path: str = DEFAULT_VECTOR_INDEX_PATH,
    top_k: int = DEFAULT_RAG_TOP_K,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL,
    embedding_api_key: str = DEFAULT_EMBEDDING_API_KEY,
    embedding_timeout_seconds: float = DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    mock_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    chat_provider: str = DEFAULT_CHAT_PROVIDER,
    chat_model: str = DEFAULT_CHAT_MODEL,
    chat_base_url: str = DEFAULT_CHAT_BASE_URL,
    chat_api_key: str = DEFAULT_CHAT_API_KEY,
    chat_timeout_seconds: float = (
        DEFAULT_CHAT_TIMEOUT_SECONDS
    ),
    temperature: float = DEFAULT_CHAT_TEMPERATURE,
    max_tokens: int = DEFAULT_CHAT_MAX_TOKENS,
    chunk_type: Optional[str] = None,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    dimension: Optional[int] = None,
) -> Dict:
    """
    执行完整基础 RAG 问答流程。
    """
    if dimension is not None:
        mock_dimension = dimension

    retrieval_result = (
        search_vector_index_from_file(
            query=query,
            index_path=index_path,
            top_k=top_k,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            timeout_seconds=embedding_timeout_seconds,
            mock_dimension=mock_dimension,
            chunk_type=chunk_type,
            include_content=True,
        )
    )

    retrieved_chunks = retrieval_result[
        "results"
    ]

    messages = build_rag_messages(
        query=query,
        retrieved_chunks=retrieved_chunks,
        max_context_chars=max_context_chars,
    )

    answer = generate_chat_response(
        messages=messages,
        provider=chat_provider,
        model_name=chat_model,
        base_url=chat_base_url,
        api_key=chat_api_key,
        timeout_seconds=chat_timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    citations = build_citations(
        retrieved_chunks
    )

    answer_quality = evaluate_answer_quality(
        answer=answer,
        citations=citations,
    )

    return {
        "query": query,
        "answer": answer,
        "chat_provider": chat_provider,
        "chat_model": chat_model,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "dimension": retrieval_result["dimension"],
        "top_k": top_k,
        "chunk_type": chunk_type,
        "retrieval_count": len(
            retrieved_chunks
        ),
        "citations": citations,
        "answer_quality": answer_quality,
    }
