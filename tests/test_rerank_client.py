from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from rerank_client import (
    RerankClient,
    RerankConfig,
    RerankServiceError,
    clear_rerank_client_cache,
    get_cached_rerank_client,
)


class FakeCrossEncoder:
    """
    模拟 CrossEncoder，避免单元测试加载真实模型。
    """

    def predict(
        self,
        pairs,
        batch_size,
        show_progress_bar,
    ):
        assert pairs == [
            ["query", "document one"],
            ["query", "document two"],
        ]
        assert batch_size == 4
        assert show_progress_bar is False

        return [0.2, 0.9]


class BrokenCrossEncoder:
    """
    模拟真实 CrossEncoder 推理失败。
    """

    def predict(
        self,
        pairs,
        batch_size,
        show_progress_bar,
    ):
        raise RuntimeError("predict failed")


def test_mock_rerank_scores_more_relevant_document_higher():
    client = RerankClient(
        config=RerankConfig(
            provider="mock",
            model_name_or_path="mock",
        )
    )

    scores = client.score(
        query="EmbeddingClient",
        documents=[
            "EmbeddingClient 负责生成向量",
            "数据库包含 projects 表",
        ],
    )

    assert scores[0] > scores[1]


def test_cross_encoder_injection_uses_fake_model():
    client = RerankClient(
        config=RerankConfig(
            provider="sentence_transformers",
            model_name_or_path="fake-model",
            batch_size=4,
        ),
        model=FakeCrossEncoder(),
    )

    scores = client.score(
        query="query",
        documents=[
            "document one",
            "document two",
        ],
    )

    assert scores == [0.2, 0.9]


def test_cross_encoder_error_is_wrapped_as_rerank_service_error():
    client = RerankClient(
        config=RerankConfig(
            provider="sentence_transformers",
            model_name_or_path="fake-model",
        ),
        model=BrokenCrossEncoder(),
    )

    with pytest.raises(RerankServiceError):
        client.score(
            query="query",
            documents=["document"],
        )


def test_get_cached_rerank_client_reuses_same_config():
    clear_rerank_client_cache()
    config = RerankConfig(
        provider="mock",
        model_name_or_path="mock",
    )

    first_client = get_cached_rerank_client(config)
    second_client = get_cached_rerank_client(config)

    assert first_client is second_client


def test_get_cached_rerank_client_uses_different_key_for_different_model():
    clear_rerank_client_cache()
    first_client = get_cached_rerank_client(
        RerankConfig(
            provider="mock",
            model_name_or_path="mock-a",
        )
    )
    second_client = get_cached_rerank_client(
        RerankConfig(
            provider="mock",
            model_name_or_path="mock-b",
        )
    )

    assert first_client is not second_client


@pytest.mark.parametrize(
    "config",
    [
        RerankConfig(provider="bad", model_name_or_path="mock"),
        RerankConfig(provider="mock", model_name_or_path=" "),
        RerankConfig(provider="mock", model_name_or_path="mock", batch_size=0),
        RerankConfig(provider="mock", model_name_or_path="mock", max_length=0),
    ],
)
def test_rerank_config_rejects_invalid_values(config):
    with pytest.raises(ValueError):
        config.validate()


def test_score_rejects_empty_query():
    client = RerankClient()

    with pytest.raises(ValueError):
        client.score(" ", ["document"])


def test_score_returns_empty_list_for_no_documents():
    client = RerankClient()

    assert client.score("query", []) == []


def test_score_rejects_non_string_document():
    client = RerankClient()

    with pytest.raises(TypeError):
        client.score("query", ["document", 123])


def test_score_rejects_empty_document():
    client = RerankClient()

    with pytest.raises(ValueError):
        client.score("query", [" "])
