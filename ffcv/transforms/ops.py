"""
General operations:

- Collation
- Conversion to PyTorch Tensor
"""
import torch as ch
import numpy as np
from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.stage import Stage
from ffcv.pipeline.state import State
from dataclasses import replace

class Collate(Operation):
    def __init__(self):
        super().__init__()
    
    def generate_code(self) -> Callable:
        def collate(image, *_):
            return image
        return collate
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.stage == Stage.INDIVIDUAL
        return replace(previous_state, stage=Stage.BATCHES), None

class ToTensor(Operation):
    def __init__(self):
        super().__init__()
    
    def generate_code(self) -> Callable:
        def to_tensor(image, dst, _):
            import ipdb; ipdb.set_trace()
            dst[:] = np.transpose(image, [0, 1 ,2])
            return ch.from_numpy(image)
        return to_tensor
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.stage == Stage.BATCHES
        return previous_state, AllocationQuery((3, 32, 32), dtype=np.dtype('uint8'))
