from .utils import fast_crop
import numpy as np
from typing import Callable, Optional, Tuple
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.stage import Stage
from ..pipeline.state import State

class RandomResizedCrop(Operation):
    def __init__(self, scale: Tuple[float, float], ratio: Tuple[float, float], size: int):
        super().__init__()
        self.scale = scale
        self.ratio = ratio
        self.size = size
    
    def generate_code(self) -> Callable:
        def random_resized_crop(im, dst, _):
            i, j, h, w = fast_crop.get_random_crop(im.shape[0], 
                                                im.shape[1],
                                                self.scale,
                                                self.ratio)
            fast_crop.resize_crop(im, i, i + h, j, j + w, dst)
            return dst

        return random_resized_crop
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        assert previous_state.stage == Stage.INDIVIDUAL
        return previous_state, AllocationQuery((self.size, self.size, 3), dtype=np.dtype('uint8'))


