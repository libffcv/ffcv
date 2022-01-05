"""
Masked applied on a predefined set of images
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
        Cutout parameter alpha
    """

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def generate_code(self) -> Callable:
        alpha = self.alpha
        my_range = Compiler.get_iterator()

        def mixer(images, temp_array, indices):
            print('IMAGES 1', images[0][0][0])
            num_images = images.shape[0]
            # permutation = np.random.permutation(num_images)
            permutation = np.argsort(indices)
            lam = np.zeros((num_images,))
            with objmode(lam="float32[:]"):
                rng = np.random.default_rng(indices[-1])
                lam = rng.beta(alpha, alpha, size=num_images).astype(np.float32)

            for ix in my_range(num_images):
                temp_array[ix] = lam[ix] * images[ix] + (1 - lam[ix]) * images[permutation[ix]]

            print(lam)
            images[:] = temp_array
            return images

        mixer.is_parallel = True
        mixer.with_indices = True

        return mixer

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        # assert previous_state.jit_mode
        # We do everything in place
        return (previous_state, AllocationQuery(shape=previous_state.shape,
                                                dtype=np.float32))

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
            num_labels = labels.shape[0]
            # permutation = np.random.permutation(num_labels)
            permutation = np.argsort(indices)
            with objmode(lam="float32[:]"):
                rng = np.random.default_rng(indices[-1])
                lam = rng.beta(alpha, alpha, size=num_labels).astype(np.float32)

            for ix in my_range(num_labels):
                temp_array[ix, 0] = labels[ix][0]
                temp_array[ix, 1] = labels[permutation[ix]][0]
                temp_array[ix, 2] = lam[ix]

            return temp_array

        mixer.is_parallel = True
        mixer.with_indices = True

        return mixer

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        # assert previous_state.jit_mode
        # We do everything in place
        return (replace(previous_state, shape=(3,), dtype=np.float32), 
                AllocationQuery((3,), dtype=np.float32))

class MixupToOneHot(Operation):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
    
    def generate_code(self) -> Callable:
        C = self.num_classes
        def one_hotter(mixedup_labels, dst):
            # Disgusting in-place version:
            # L * a + (1-L) * b = L * (a + (1/L - 1) * b)
            mixedup_labels[:,2].reciprocal_()
            mixedup_labels[:,2] -= 1.
            dst[:] = F.one_hot(mixedup_labels[:,1].long(), C)
            dst *= mixedup_labels[:,2][:,None]
            dst += mixedup_labels[:,0][:,None]
            mixedup_labels[:,2] += 1
            mixedup_labels[:,2].reciprocal_()
            dst *= mixedup_labels[:,2][:,None]
            return dst
        
        return one_hotter

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert not previous_state.jit_mode
        return (replace(previous_state, shape=(self.num_classes,)), \
                AllocationQuery((self.num_classes,), dtype=previous_state.dtype, device=previous_state.device))