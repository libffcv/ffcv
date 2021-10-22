"""
Random horizontal flip
"""
from numpy import dtype
from numpy.random import rand
from typing import Callable, Optional, Tuple
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.stage import Stage
from ..pipeline.state import State

class RandomHorizontalFlip(Operation):
    def __init__(self, p: float):
        super().__init__()
        self.p = float
    
    def generate_code(self) -> Callable:
        def flip(image, dst):
            should_flip = rand() > 0.5
            dst[:] = image[:,::-1] if should_flip else image
            return dst
        return flip
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        assert previous_state.stage == Stage.INDIVIDUAL
        return (previous_state, AllocationQuery((32, 32, 3), dtype('uint8')))