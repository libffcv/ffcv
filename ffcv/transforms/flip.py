"""
Random horizontal flip
"""
from numpy import dtype
from numpy.random import rand
from typing import Callable, Optional, Tuple
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.compiler import Compiler

class RandomHorizontalFlip(Operation):
    """Flips the image horizontally with probability p.
    Operates on raw arrays (not tensors).

    Parameters
    ----------
    p : float
        The probability with which to flip each image in the batch
        horizontally.
    """

    def __init__(self):
        super().__init__()

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()

        def flip(images, dst):
            should_flip = rand(images.shape[0]) > 0.5
            for i in my_range(images.shape[0]):
                if should_flip[i]:
                    dst[i] = images[i, :, ::-1]
                else:
                    dst[i] = images[i]

            return dst

        flip.is_parallel = True
        return flip

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        return (previous_state, AllocationQuery(previous_state.shape, previous_state.dtype))
