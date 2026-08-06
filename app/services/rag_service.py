from typing import Dict, Optional

from services.answer_quality import evaluate_answer_quality
from services.citation_builder import build_citations
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
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_CANDIDATE_TOP_K,
    DEFAULT_RERANK_DEVICE,
    DEFAULT_RERANK_LOCAL_FILES_ONLY,
    DEFAULT_RERANK_MAX_LENGTH,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_PROVIDER,
    DEFAULT_VECTOR_INDEX_PATH,
)
from services.hybrid_search_service import hybrid_search_from_files
from clients.llm_client import generate_chat_response
from services.prompt_builder import build_rag_messages
from pipelines.retrieval_pipeline import retrieve_with_rerank
from services.vector_search_service import (
    search_vector_index_from_file,
)
from context_engineering.secure_context_builder import SecureContextBuilder


def ask_from_vector_index(
    query: str,
    retrieval_mode: str = "vector",
    chunks_path: str = "outputs/chunks.json",
    index_path: str = DEFAULT_VECTOR_INDEX_PATH,
    top_k: int = DEFAULT_RAG_TOP_K,
    candidate_top_k: int = DEFAULT_RERANK_CANDIDATE_TOP_K,
    keyword_weight: float = 0.4,
    vector_weight: float = 0.6,
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
    rerank_provider: str = DEFAULT_RERANK_PROVIDER,
    rerank_model: str = DEFAULT_RERANK_MODEL,
    rerank_device: str = DEFAULT_RERANK_DEVICE,
    rerank_batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    rerank_max_length: int = DEFAULT_RERANK_MAX_LENGTH,
    rerank_local_files_only: bool = DEFAULT_RERANK_LOCAL_FILES_ONLY,
    chunk_type: Optional[str] = None,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    dimension: Optional[int] = None,
    query_strategy: str = "original",
    rewrite_count: int = 2,
) -> Dict:
    """
    执行完整基础 RAG 问答流程。
    """
    if dimension is not None:
        mock_dimension = dimension

    if retrieval_mode == "rerank":
        retrieval_result = retrieve_with_rerank(
            query=query,
            chunks_path=chunks_path,
            index_path=index_path,
            candidate_top_k=candidate_top_k,
            final_top_k=top_k,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            embedding_timeout_seconds=embedding_timeout_seconds,
            mock_dimension=mock_dimension,
            rerank_provider=rerank_provider,
            rerank_model=rerank_model,
            rerank_device=rerank_device,
            rerank_batch_size=rerank_batch_size,
            rerank_max_length=rerank_max_length,
            rerank_local_files_only=rerank_local_files_only,
            chunk_type=chunk_type,
            query_strategy=query_strategy,
            rewrite_count=rewrite_count,
            query_rewrite_provider=chat_provider,
            query_rewrite_model=chat_model,
            query_rewrite_base_url=chat_base_url,
            query_rewrite_api_key=chat_api_key,
            query_rewrite_timeout_seconds=chat_timeout_seconds,
        )

    elif retrieval_mode == "hybrid":
        retrieval_result = hybrid_search_from_files(
            query=query,
            chunks_path=chunks_path,
            index_path=index_path,
            keyword_top_k=candidate_top_k,
            vector_top_k=candidate_top_k,
            final_top_k=top_k,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            embedding_timeout_seconds=embedding_timeout_seconds,
            mock_dimension=mock_dimension,
            chunk_type=chunk_type,
        )

    elif retrieval_mode == "vector":
        retrieval_result = search_vector_index_from_file(
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

    else:
        raise ValueError(f"不支持的 retrieval_mode：{retrieval_mode}")

    retrieved_chunks = retrieval_result[
        "results"
    ]

    secure_context = SecureContextBuilder().build(
        evidence_items=retrieved_chunks
    )
    retrieved_chunks = secure_context.selected_evidence
    if not retrieved_chunks:
        raise ValueError("安全上下文过滤后没有可用检索证据")

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
        "dimension": retrieval_result.get("dimension"),
        "retrieval_mode": retrieval_mode,
        "query_strategy": query_strategy,
        "rewrite_count": rewrite_count,
        "query_items": retrieval_result.get("query_items"),
        "rewrite_result": retrieval_result.get("rewrite_result"),
        "top_k": top_k,
        "candidate_top_k": candidate_top_k,
        "chunk_type": chunk_type,
        "retrieval_count": len(
            retrieved_chunks
        ),
        "citations": citations,
        "answer_quality": answer_quality,
    }
