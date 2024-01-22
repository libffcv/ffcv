"""
Random Solarization
"""
from numpy.random import rand
from typing import Callable, Optional, Tuple
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.compiler import Compiler

class Solarization(Operation):
    """Solarize the image randomly with a given probability by inverting all pixel
    values above a threshold. If img is a Tensor, it is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Parameters
    ----------
        solarization_prob (float): probability of the image being solarized. Default value is 0.5
        threshold (float): all pixels equal or above this value are inverted.
    """

    def __init__(self, solarization_prob: float = 0.5, threshold: float = 128):
        super().__init__()
        self.solarization_prob = solarization_prob
        self.threshold = threshold

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        solarization_prob = self.solarization_prob
        threshold = self.threshold

        def solarize(images, dst):
            should_solarize = rand(images.shape[0]) < solarization_prob
            for i in my_range(images.shape[0]):
                if should_solarize[i]:
                    mask = (images[i] >= threshold) 
                    dst[i] = images[i] * (1-mask) + (255 - images[i])*mask
                else:
                    dst[i] = images[i]
            return dst

        solarize.is_parallel = True
        return solarize

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        return (previous_state, AllocationQuery(previous_state.shape, previous_state.dtype))
