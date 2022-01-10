"""
Random crop
"""
import numpy as np
from numpy import dtype
from numpy.random import randint
from typing import Any, Callable, Optional, Tuple, Union
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.compiler import Compiler

class RandomTranslate(Operation):
    """Flips the image horizontally with probability p.
    Operates on raw arrays (not tensors).

    Parameters
    ----------
    padding : int
        The probability with which to flip each image in the batch
        horizontally.
    """

    def __init__(self, padding: int):
        super().__init__()
        self.padding = padding

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        pad = self.padding

        def translate(images, dst):
            n, h, w, _ = images.shape
            dst[:, pad:pad+h, pad:pad+w] = images
            y_coords = randint(low=0, high=2 * pad + 1, size=(n,))
            x_coords = randint(low=0, high=2 * pad + 1, size=(n,))
            for i in my_range(n):
                images[i] = dst[i, y_coords[i]:y_coords[i]+h, x_coords[i]:x_coords[i]+w]

            return images

        translate.is_parallel = True
        return translate

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        h, w, c = previous_state.shape
        assert previous_state.jit_mode
        return (previous_state, AllocationQuery((h + 2 * self.padding, w + 2 * self.padding, c), previous_state.dtype))
