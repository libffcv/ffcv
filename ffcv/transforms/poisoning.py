"""
Poison images by adding a mask
"""
from typing import Tuple
from dataclasses import replace

import numpy as np
from typing import Callable, Optional, Tuple
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.compiler import Compiler

class Poison(Operation):
    """Poison specified images by adding a mask with given opacity.
    Operates on raw arrays (not tensors).

    Parameters
    ----------
    mask : ndarray
        The mask to apply to each image.
    alpha: float
        The opacity of the mask.
    indices : Sequence[int]
        The indices of images that should have the mask applied.
    clamp : Tuple[int, int]
        Clamps the final pixel values between these two values (default: (0, 255)).
    """

    def __init__(self, mask: np.ndarray, alpha: np.ndarray,
                 indices, clamp = (0, 255)):
        super().__init__()
        self.mask = mask
        self.indices = np.sort(indices)
        self.clamp = clamp
        self.alpha = alpha

    def generate_code(self) -> Callable:

        alpha = np.repeat(self.alpha[:, :, None], 3, axis=2)
        mask = self.mask.astype('float') * alpha
        to_poison = self.indices
        clamp = self.clamp
        my_range = Compiler.get_iterator()

        def poison(images, temp_array, indices):
            for i in my_range(images.shape[0]):
                sample_ix = indices[i]
                # We check if the index is in the list of indices
                # to poison
                position = np.searchsorted(to_poison, sample_ix)
                if position < len(to_poison) and to_poison[position] == sample_ix:
                    temp = temp_array[i]
                    temp[:] = images[i]
                    temp *= 1 - alpha
                    temp += mask
                    np.clip(temp, clamp[0], clamp[1], out=temp)
                    images[i] = temp
            return images

        poison.is_parallel = True
        poison.with_indices = True

        return poison

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        # We do everything in place
        return (replace(previous_state, jit_mode=True), \
                AllocationQuery(shape=previous_state.shape, dtype=np.dtype('float32')))
