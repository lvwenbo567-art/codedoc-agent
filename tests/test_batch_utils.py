from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "app"))

from utils.batch_utils import iter_batches, split_batches


def test_split_batches():
    result = split_batches(
        [1, 2, 3, 4, 5],
        batch_size=2,
    )

    assert result == [
        [1, 2],
        [3, 4],
        [5],
    ]


def test_split_empty_list():
    assert split_batches(
        [],
        batch_size=2,
    ) == []


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        split_batches(
            [1, 2],
            batch_size=0,
        )


def test_iter_batches_is_generator():
    generator = iter_batches(
        [1, 2, 3, 4, 5],
        batch_size=2,
    )

    assert iter(generator) is generator
    assert next(generator) == [1, 2]
    assert next(generator) == [3, 4]
    assert next(generator) == [5]

    with pytest.raises(StopIteration):
        next(generator)
