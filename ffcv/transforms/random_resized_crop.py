"""
Random resized crop, similar to torchvision.transforms.RandomResizedCrop
"""
from dataclasses import replace
from .utils import fast_crop
import numpy as np
from typing import Callable, Optional, Tuple
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State

class RandomResizedCrop(Operation):
    """Crop a random portion of image with random aspect ratio and resize it to a given size.

    Parameters
    ----------
    scale : Tuple[float, float]
        Lower and upper bounds for the ratio of random area of the crop.
    ratio : Tuple[float, float]
        Lower and upper bounds for random aspect ratio of the crop.
    size : int
        Side length of the output.
    """
    def __init__(self, scale: Tuple[float, float], ratio: Tuple[float, float], size: int):
        super().__init__()
        self.scale = scale
        self.ratio = ratio
        self.size = size

    def generate_code(self) -> Callable:
        scale, ratio = self.scale, self.ratio
        def random_resized_crop(im, dst):
            i, j, h, w = fast_crop.get_random_crop(im.shape[0],
                                                im.shape[1],
                                                scale,
                                                ratio)
            fast_crop.resize_crop(im, i, i + h, j, j + w, dst)
            return dst

        return random_resized_crop

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        return replace(previous_state, shape=(self.size, self.size, 3)), AllocationQuery((self.size, self.size, 3), dtype=np.dtype('uint8'))


