"""
Mixup augmentation [CITE]
"""
import numpy as np
from numpy.random import permutation, rand
from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.stage import Stage
from ffcv.pipeline.state import State

class LabelMixup(Operation):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha
    
    def generate_code(self) -> Callable:
        def label_mixup(lab_batch, dst):
            mixup_order = permutation(lab_batch.shape[0])
            lam = rand(lab_batch.shape[0])
            dst[:,0] = lab_batch
            dst[:,1] = lab_batch[mixup_order]
            dst[:,2] = lam
            return dst
            
        return label_mixup
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return previous_state, AllocationQuery((3,), dtype=np.dtype('float16'))

class Mixup(Operation):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha
    
    def generate_code(self) -> Callable:
        def mixup_batch(image_batch, dst):
            # Generate image permtuation and fill dst
            mixup_order = permutation(image_batch.shape[0])
            dst[:] = image_batch[mixup_order]

            # Generate mixing probabilities, also with seed
            lam = rand(image_batch.shape[0], 1, 1, 1)
            
            # Perform mixup
            image_batch *= lam
            dst *= (1. - lam)
            image_batch += dst
            return image_batch

        return mixup_batch
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        assert previous_state.stage == Stage.BATCHES
        # TODO: what to do with dtype?
        return previous_state, AllocationQuery(previous_state.shape)
