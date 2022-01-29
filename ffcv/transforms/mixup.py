"""
Mixup augmentation for images and labels (https://arxiv.org/abs/1710.09412)
"""
from typing import Tuple

from numba import objmode
import numpy as np
import torch as ch
import torch.nn.functional as F
from dataclasses import replace
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
        Mixup parameter alpha
    same_lambda : bool
        Whether to use the same value of lambda across the whole batch, or an
        individually sampled lambda per image in the batch
    """

    def __init__(self, alpha: float, same_lambda: bool):
        super().__init__()
        self.alpha = alpha
        self.same_lambda = same_lambda

    def generate_code(self) -> Callable:
        alpha = self.alpha
        same_lam = self.same_lambda
        my_range = Compiler.get_iterator()

        def mixer(images, dst, indices):
            np.random.seed(indices[-1])
            num_images = images.shape[0]
            lam = np.random.beta(alpha, alpha) if same_lam else \
                  np.random.beta(alpha, alpha, num_images)
            for ix in my_range(num_images):
                l = lam if same_lam else lam[ix]
                dst[ix] = l * images[ix] + (1 - l) * images[ix - 1]

            return dst

        mixer.is_parallel = True
        mixer.with_indices = True

        return mixer

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (previous_state, AllocationQuery(shape=previous_state.shape,
                                                dtype=previous_state.dtype))

class LabelMixup(Operation):
    """Mixup for labels. Should be initialized in exactly the same way as
    :cla:`ffcv.transforms.ImageMixup`.
    """
    def __init__(self, alpha: float, same_lambda: bool):
        super().__init__()
        self.alpha = alpha
        self.same_lambda = same_lambda

    def generate_code(self) -> Callable:
        alpha = self.alpha
        same_lam = self.same_lambda
        my_range = Compiler.get_iterator()

        def mixer(labels, temp_array, indices):
            num_labels = labels.shape[0]
            # permutation = np.random.permutation(num_labels)
            np.random.seed(indices[-1])
            lam = np.random.beta(alpha, alpha) if same_lam else \
                  np.random.beta(alpha, alpha, num_labels)

            for ix in my_range(num_labels):
                temp_array[ix, 0] = labels[ix][0]
                temp_array[ix, 1] = labels[ix - 1][0]
                temp_array[ix, 2] = lam if same_lam else lam[ix]

            return temp_array

        mixer.is_parallel = True
        mixer.with_indices = True

        return mixer

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, shape=(3,), dtype=np.float32),
                AllocationQuery((3,), dtype=np.float32))

class MixupToOneHot(Operation):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def generate_code(self) -> Callable:
        def one_hotter(mixedup_labels, dst):
            dst.zero_()
            N = mixedup_labels.shape[0]
            dst[ch.arange(N), mixedup_labels[:, 0].long()] = mixedup_labels[:, 2]
            mixedup_labels[:, 2] *= -1
            mixedup_labels[:, 2] += 1
            dst[ch.arange(N), mixedup_labels[:, 1].long()] = mixedup_labels[:, 2]
            return dst

        return one_hotter

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        # Should already be converted to tensor
        assert not previous_state.jit_mode
        return (replace(previous_state, shape=(self.num_classes,)), \
                AllocationQuery((self.num_classes,), dtype=previous_state.dtype, device=previous_state.device))