"""
Cutout augmentation (https://arxiv.org/abs/1708.04552)
"""
import numpy as np
from typing import Callable, Optional, Tuple
from dataclasses import replace

from ffcv.pipeline.compiler import Compiler
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State

class Cutout(Operation):
    """Cutout data augmentation (https://arxiv.org/abs/1708.04552).

    Parameters
    ----------
    crop_size : int
        Size of the random square to cut out.
    fill : Tuple[int, int, int], optional
        An RGB color ((0, 0, 0) by default) to fill the cutout square with.
        Useful for when a normalization layer follows cutout, in which case
        you can set the fill such that the square is zero
        post-normalization.
    """
    def __init__(self, crop_size: int, fill: Tuple[int, int, int] = (0, 0, 0)):
        super().__init__()
        self.crop_size = crop_size
        self.fill = np.array(fill)

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        crop_size = self.crop_size
        fill = self.fill

        def cutout_square(images, *_):
            for i in my_range(images.shape[0]):
                # Generate random origin
                coord = (
                    np.random.randint(images.shape[1] - crop_size + 1),
                    np.random.randint(images.shape[2] - crop_size + 1),
                )
                # Black out image in-place
                images[i, coord[0]:coord[0] + crop_size, coord[1]:coord[1] + crop_size] = fill

            return images
        cutout_square.is_parallel = True

        return cutout_square

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return replace(previous_state, jit_mode=True), None
