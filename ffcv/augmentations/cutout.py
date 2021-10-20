"""
Cutout augmentation [CITE]
"""
from numpy.random import randint
from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.stage import Stage
from ffcv.pipeline.state import State

class Cutout(Operation):
    def __init__(self, crop_size: int):
        super().__init__()
        self.crop_size = crop_size
    
    def generate_code(self) -> Callable:
        def cutout_square(state: State, image, dst):
            # Generate random origin
            x_src, y_src = randint(high=image.shape[-1], shape=(2,))
            # Black out image in-place
            image[:, :, y_src:y_src + self.crop_size, x_src:x_src + self.crop_size] = 0
            return image

        return cutout_square
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        assert previous_state.stage == Stage.INDIVIDUAL
        return previous_state, None