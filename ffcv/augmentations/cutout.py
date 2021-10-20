"""
Cutout augmentation [CITE]
"""
import numba
from numpy.random import randint
from typing import Callable
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.stage import Stage
from ffcv.pipeline.state import State

class Cutout(Operation):
    def __init__(self, crop_size: int):
        super().__init__()
        self.crop_size = crop_size
    
    def generate_code(self) -> Callable:
        @numba.jit
        def cutout_square(state: State, image, dst):
            assert state.jit_mode, (state.stage == Stage.INDIVIDUAL)
            # Generate random origin
            x_src, y_src = randint(high=image.shape[-1], shape=(2,))
            # Black out image in-place
            image[:, :, y_src:y_src + self.crop_size, x_src:x_src + self.crop_size] = 0
            return image

        return cutout_square
    
    def advance_state(self, previous_state: State) -> State:
        return previous_state