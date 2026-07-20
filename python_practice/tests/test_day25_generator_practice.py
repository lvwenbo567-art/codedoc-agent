from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from day25_generator_practice import (
    calculate_total_length,
    filter_chunk_contents,
    iter_chunk_batches,
    iter_file_lines,
)


def build_chunks():
    return [
        {
            "chunk_type": "code",
            "content": "def main(): pass",
            "length": 16,
        },
        {
            "chunk_type": "document",
            "content": "项目说明",
            "length": 4,
        },
        {
            "chunk_type": "code",
            "content": "class App: pass",
            "length": 15,
        },
    ]


def test_iter_file_lines(tmp_path):
    file_path = tmp_path / "demo.txt"
    file_path.write_text("a\nb\nc\n", encoding="utf-8")

    generator = iter_file_lines(str(file_path))

    assert iter(generator) is generator
    assert list(generator) == ["a", "b", "c"]


def test_iter_file_lines_not_exists(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_file_lines(str(tmp_path / "missing.txt")))


def test_iter_file_lines_path_is_not_file(tmp_path):
    with pytest.raises(ValueError):
        list(iter_file_lines(str(tmp_path)))


def test_iter_chunk_batches():
    generator = iter_chunk_batches(
        build_chunks(),
        batch_size=2,
    )

    assert iter(generator) is generator
    assert len(next(generator)) == 2
    assert len(next(generator)) == 1

    with pytest.raises(StopIteration):
        next(generator)


def test_iter_chunk_batches_invalid_batch_size():
    with pytest.raises(ValueError):
        list(
            iter_chunk_batches(
                build_chunks(),
                batch_size=0,
            )
        )


def test_filter_chunk_contents():
    assert filter_chunk_contents(
        build_chunks(),
        chunk_type="code",
    ) == [
        "def main(): pass",
        "class App: pass",
    ]


def test_calculate_total_length():
    assert calculate_total_length(build_chunks()) == 35
