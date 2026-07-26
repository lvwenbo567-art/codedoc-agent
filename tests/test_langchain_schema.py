from pathlib import Path
import sys


sys.path.append(
    str(
        Path(__file__).resolve().parents[1] / "app"
    )
)


from schemas.langchain_schema import LangChainAgentRequest


def test_langchain_agent_request_accepts_retrieval_model_config():
    request = LangChainAgentRequest(
        query="keyword_score 在哪里定义？",
        chunks_path="outputs/test_project_chunks.json",
        index_path="outputs/test_project_vector_index_bge_m3.json",
        embedding_provider="ollama",
        embedding_model="bge-m3",
        embedding_base_url="http://localhost:11434",
        embedding_api_key="",
        embedding_timeout_seconds=120,
        mock_dimension=1024,
        rerank_provider="sentence_transformers",
        rerank_model="D:/models/bge-reranker-v2-m3",
        rerank_device="cpu",
        rerank_batch_size=8,
        rerank_max_length=512,
        rerank_local_files_only=True,
        project_id=3,
        thread_id="day34-thread",
        user_id="student-1",
        run_id="run-manual",
        trace_id="trace-manual",
    )

    assert request.embedding_provider == "ollama"
    assert request.embedding_model == "bge-m3"
    assert request.rerank_provider == "sentence_transformers"
    assert request.rerank_local_files_only is True
    assert request.project_id == 3
    assert request.thread_id == "day34-thread"
    assert request.user_id == "student-1"
    assert request.run_id == "run-manual"
    assert request.trace_id == "trace-manual"
