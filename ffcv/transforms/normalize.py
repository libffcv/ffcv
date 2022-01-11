"""
Masked applied on a predefined set of images
"""
from collections.abc import Sequence
from typing import Tuple

import numpy as np
from numpy import dtype
from numpy.random import rand
from dataclasses import replace
from typing import Callable, Optional, Tuple
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.compiler import Compiler

class NormalizeImage(Operation):
    """Perform Image Normalization.
    Operates on raw arrays (not tensors).

    Parameters
    ----------
    mean: np.ndarray
        The mean vector
    std: np.ndarray
        The standard deviation
    type: np.dtype
        The desired output type for the result
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray,
                 type: np.dtype):
        super().__init__()
        table = (np.arange(256)[:, None] - mean[None, :]) / std[None, :]
        table = table.astype(type)
        self.lookup_table = table
        if type == np.float16:
            type = np.int16
        self.dtype = type
        self.previous_shape = None

    def generate_code(self) -> Callable:

        table = self.lookup_table.view(dtype=self.dtype)
        my_range = Compiler.get_iterator()
        previous_shape = self.previous_shape

        def normalize_convert(images, result, indices):
            result_flat = result.reshape(result.shape[0], -1, 3)
            num_pixels = result_flat.shape[1]
            for i in my_range(len(indices)):
                image = images[i].reshape(num_pixels, 3)
                for px in range(num_pixels):
                    # Just in case llvm forgets to unroll this one
                    result_flat[i, px, 0] = table[image[px, 0], 0]
                    result_flat[i, px, 1] = table[image[px, 1], 0]
                    result_flat[i, px, 2] = table[image[px, 2], 0]
            return result

        normalize_convert.is_parallel = True
        normalize_convert.with_indices = True
        return normalize_convert

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        self.previous_shape = previous_state.shape
        my_state = (replace(previous_state, dtype=self.dtype), AllocationQuery(shape=previous_state.shape, dtype=self.dtype))

        return my_state