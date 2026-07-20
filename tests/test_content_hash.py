from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from content_hash import compute_content_hash, normalize_content


def test_same_content_same_hash():
    assert compute_content_hash("hello") == compute_content_hash("hello")


def test_different_content_different_hash():
    assert compute_content_hash("hello") != compute_content_hash("world")


def test_windows_and_linux_newline_same_hash():
    assert compute_content_hash("hello\r\nworld") == compute_content_hash(
        "hello\nworld"
    )


def test_normalize_content_rejects_non_string():
    with pytest.raises(TypeError):
        normalize_content(123)
