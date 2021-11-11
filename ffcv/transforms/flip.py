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
    def __init__(self, p: float):
        super().__init__()
        self.p = float

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()

        def flip(images, dst):
            for i in my_range(images.shape[0]):
                should_flip = rand() > 0.5
                if should_flip:
                    dst[i] = images[i, :, ::-1]
            return dst

        flip.is_parallel = True

        return flip

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        return (previous_state, AllocationQuery(previous_state.shape, previous_state.dtype))
