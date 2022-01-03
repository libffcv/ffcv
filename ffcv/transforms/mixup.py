"""
Masked applied on a predefined set of images
"""
from typing import Tuple

import numpy as np
import torch as ch
from typing import Callable, Optional, Tuple
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.compiler import Compiler

class ImageMixup(Operation):
    """Mixup for images. Operates on raw arrays (not tensors).

    Parameters
    ----------
    alpha : float
        Cutout parameter alpha
    """

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def generate_code(self) -> Callable:
        alpha = self.alpha
        my_range = Compiler.get_iterator()

        def mixer(images, temp_array, indices):
            rng = np.random.default_rng(indices[0])
            num_images = images.shape[0]
            permutation = rng.permutation(num_images)
            lam = rng.beta(alpha, alpha)

            for ix in my_range(num_images):
                temp_array[ix] = images[permutation[ix]]

            images[:] = images * lam + temp_array * (1 - lam)

            return images

        mixer.is_parallel = True
        mixer.with_indices = True

        return mixer

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        # assert previous_state.jit_mode
        # We do everything in place
        return (previous_state, AllocationQuery(shape=previous_state.shape,
                                                dtype=ch.float32))

class LabelMixup(Operation):
    """Cutout for labels.

    Parameters
    ----------
    alpha : float
        Cutout parameter alpha
    """
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def generate_code(self) -> Callable:
        alpha = self.alpha
        my_range = Compiler.get_iterator()

        def mixer(labels, temp_array, indices):
            rng = np.random.default_rng(indices[0])
            num_labels = labels.shape[0]
            permutation = rng.permutation(num_labels)
            lam = rng.beta(alpha, alpha)

            for ix in my_range(num_labels):
                temp_array[ix, 0] = labels[ix]
                temp_array[ix, 1] = labels[permutation[ix]]
                temp_array[ix, 2] = lam

            return temp_array

        mixer.is_parallel = True
        mixer.with_indices = True

        return mixer

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        # assert previous_state.jit_mode
        # We do everything in place
        return (previous_state, AllocationQuery((3,), dtype=ch.float32))
