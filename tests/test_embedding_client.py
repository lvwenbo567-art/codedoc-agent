from pathlib import Path
import math
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from embedding_client import EmbeddingClient


def test_embed_text_dimension():
    client = EmbeddingClient(dimension=32)

    embedding = client.embed_text("def main(): pass")

    assert len(embedding) == 32


def test_embed_text_is_deterministic():
    client = EmbeddingClient(dimension=32)

    embedding_1 = client.embed_text("hello world")
    embedding_2 = client.embed_text("hello world")

    assert embedding_1 == embedding_2


def test_embed_text_is_normalized():
    client = EmbeddingClient(dimension=32)

    embedding = client.embed_text("hello world")

    norm = math.sqrt(sum(value * value for value in embedding))

    assert abs(norm - 1.0) < 1e-6


def test_embed_text_empty():
    client = EmbeddingClient(dimension=32)

    with pytest.raises(ValueError):
        client.embed_text("   ")


def test_invalid_dimension():
    with pytest.raises(ValueError):
        EmbeddingClient(dimension=0)