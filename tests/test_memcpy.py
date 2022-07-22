import numpy as np

import pytest
from assertpy import assert_that

from ffcv.libffcv import memcpy


MEMCPY_TYPES = [
    np.uint8,
    np.int32,
    np.int64,
    np.float64,
    np.float32
]

@pytest.mark.parametrize('dtype', MEMCPY_TYPES)
def test_memcpy(dtype):

    data = np.random.uniform(0, 255, size=(100, 99)).astype(dtype)
    dst = np.empty((100, 99), dtype=dtype)

    assert_that(np.all(data == dst)).is_false()
    memcpy(data, dst)

    assert_that(np.all(data == dst)).is_true()