"""
Random translate
"""
import numpy as np
from numpy.random import randint
from typing import Callable, Optional, Tuple
from dataclasses import replace
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.compiler import Compiler

class RandomTranslate(Operation):
    """Translate each image randomly in vertical and horizontal directions
    up to specified number of pixels.

    Parameters
    ----------
    padding : int
        Max number of pixels to translate in any direction.
    fill : tuple
        An RGB color ((0, 0, 0) by default) to fill the area outside the shifted image.
    """

    def __init__(self, padding: int, fill: Tuple[int, int, int] = (0, 0, 0)):
        super().__init__()
        self.padding = padding
        self.fill = np.array(fill)

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        pad = self.padding
        fill = self.fill

        def translate(images, dst):
            n, h, w, _ = images.shape
            dst[:] = fill
            dst[:, pad:pad+h, pad:pad+w] = images
            for i in my_range(n):
                dst[i] = 0
                y_coord = randint(low=0, high=2 * pad + 1)
                x_coord = randint(low=0, high=2 * pad + 1)
                images[i] = dst[i, y_coord:y_coord+h, x_coord:x_coord+w]

            return images

        translate.is_parallel = True
        return translate

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        h, w, c = previous_state.shape
        return (replace(previous_state, jit_mode=True), \
                AllocationQuery((h + 2 * self.padding, w + 2 * self.padding, c), previous_state.dtype))

