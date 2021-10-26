"""
Cutout augmentation [CITE]
"""
import numpy as np
from numpy.random import randint
from typing import Callable, Optional, Tuple
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State

class Cutout(Operation):
    def __init__(self, crop_size: int, fill: Tuple[int, int, int] = (0, 0, 0)):
        super().__init__()
        self.crop_size = crop_size
        self.fill = np.array(fill)
    
    def generate_code(self) -> Callable:
        crop_size = self.crop_size
        fill = self.fill
        def cutout_square(image, *_):
            # Generate random origin
            coord = (
                randint(image.shape[0] - crop_size),
                randint(image.shape[1] - crop_size),
            )
            # Black out image in-place
            image[coord[0]:coord[0] + crop_size, coord[1]:coord[1] + crop_size] = fill
            return image

        return cutout_square
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        return previous_state, None
