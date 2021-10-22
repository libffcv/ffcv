"""
Cutout augmentation [CITE]
"""
from numpy.random import randint
from typing import Callable, Optional, Tuple
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.stage import Stage
from ..pipeline.state import State

class Cutout(Operation):
    def __init__(self, crop_size: int):
        super().__init__()
        self.crop_size = crop_size
    
    def generate_code(self) -> Callable:
        def cutout_square(image, *_):
            # Generate random origin
            coord = randint(image.shape[0] - self.crop_size, size=(2,))
            # Black out image in-place
            image[coord[0]:coord[0] + self.crop_size, coord[1]:coord[1] + self.crop_size] = 0
            return image

        return cutout_square
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        assert previous_state.stage == Stage.INDIVIDUAL
        return previous_state, None