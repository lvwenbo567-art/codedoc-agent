from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))


from services.query_rewrite_service import QueryRewriteService, extract_protected_terms


class FakeConfig:
    provider = "openai_compatible"


class FakeChatClient:
    def __init__(self, response: str):
        self.response = response
        self.config = FakeConfig()

    def generate(self, messages):
        return self.response


def test_extract_protected_terms():
    query = "EmbeddingClient 在 `embedding_client.py` 中如何调用 /api/embed？"

    terms = extract_protected_terms(query)

    assert "EmbeddingClient" in terms
    assert "embedding_client.py" in terms
    assert "/api/embed" in terms


def test_rewrite_keeps_original_query_as_fallback():
    client = FakeChatClient(
        response='{"queries": ["文本向量生成流程", "向量服务调用过程"]}'
    )
    service = QueryRewriteService(chat_client=client)

    result = service.rewrite(
        query="EmbeddingClient 如何调用 /api/embed？",
        rewrite_count=2,
    )

    assert result["fallback_used"] is False

    assert result["original_query"] == "EmbeddingClient 如何调用 /api/embed？"
    assert result["rewritten_queries"][0] == "文本向量生成流程"
    assert result["rewritten_queries"][1] == "向量服务调用过程"
    assert "queries" not in result
    assert result["rewritten_query"] == "文本向量生成流程"


def test_invalid_json_falls_back():
    client = FakeChatClient(response="这不是 JSON")
    service = QueryRewriteService(chat_client=client)
    original = "EmbeddingClient 的作用是什么？"

    result = service.rewrite(query=original, rewrite_count=2)

    assert result["fallback_used"] is True
    assert result["original_query"] == original
    assert result["rewritten_queries"] == []
    assert "queries" not in result
    assert result["rewritten_query"] == original


def test_empty_model_response_uses_rule_based_fallback():
    client = FakeChatClient(response="")
    service = QueryRewriteService(chat_client=client)
    original = "这个项目是怎么把文件变成可以搜索的数据的？"

    result = service.rewrite(query=original, rewrite_count=2)

    assert result["fallback_used"] is True
    assert result["rewrite_applied"] is True
    assert len(result["rewritten_queries"]) == 2
    assert "文件扫描" in result["rewritten_queries"][0]


def test_json_code_fence_response_is_supported():
    client = FakeChatClient(
        response='```json\n{"queries": ["向量索引构建流程"]}\n```'
    )
    service = QueryRewriteService(chat_client=client)

    result = service.rewrite(
        query="build_vector_index_from_json 怎么构建索引？",
        rewrite_count=1,
    )

    assert result["fallback_used"] is False
    assert result["original_query"] == "build_vector_index_from_json 怎么构建索引？"
    assert result["rewritten_queries"][0] == "向量索引构建流程"
    assert result["rewritten_query"] == result["rewritten_queries"][0]


def test_json_object_inside_text_response_is_supported():
    client = FakeChatClient(
        response='下面是结果：\n{"queries": ["项目文件扫描 Chunk 切分 Embedding 向量索引流程"]}\n请查收。'
    )
    service = QueryRewriteService(chat_client=client)

    result = service.rewrite(
        query="这个项目是怎么把文件变成可以搜索的数据的？",
        rewrite_count=1,
    )

    assert result["fallback_used"] is False
    assert result["rewritten_queries"] == [
        "项目文件扫描 Chunk 切分 Embedding 向量索引流程"
    ]


def test_mock_rewrite_preserves_function_name():
    class MockConfig:
        provider = "mock"

    class MockClient:
        config = MockConfig()

    service = QueryRewriteService(chat_client=MockClient())

    result = service.rewrite(
        query="build_vector_index_from_json 在哪里？",
        rewrite_count=2,
    )

    assert len(result["rewritten_queries"]) == 2
    assert all(
        "build_vector_index_from_json" in query
        for query in result["rewritten_queries"]
    )
