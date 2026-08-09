from codedoc_demo.api import answer_question
from codedoc_demo.parser import split_text_into_chunks
from codedoc_demo.search import keyword_score, search_documents


def test_keyword_score_counts_query_terms() -> None:
    score = keyword_score(
        query="chunk search",
        text="chunk chunk search pipeline",
    )

    assert score == 3


def test_split_text_into_chunks_uses_overlap() -> None:
    chunks = split_text_into_chunks(
        document_id="doc-1",
        text="abcdefghij",
        chunk_size=5,
        overlap=2,
    )

    assert [chunk.content for chunk in chunks] == ["abcde", "defgh", "ghij"]


def test_search_documents_returns_top_k_results() -> None:
    chunks = split_text_into_chunks(
        document_id="doc-1",
        text="alpha beta beta gamma",
        chunk_size=100,
        overlap=0,
    )

    results = search_documents(
        query="beta",
        chunks=chunks,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["score"] == 2


def test_answer_question_uses_pipeline() -> None:
    result = answer_question("pytest tests")

    assert "usage" in result["answer"]
