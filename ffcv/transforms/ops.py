"""
General operations:

- Collation
- Conversion to PyTorch Tensor
"""
from abc import ABCMeta
import torch as ch
import numpy as np
from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.stage import Stage
from ffcv.pipeline.state import State
from dataclasses import replace

class CoreOp(Operation, metaclass=ABCMeta):
    pass

class Collate(CoreOp):
    def __init__(self):
        super().__init__()
    
    def generate_code(self) -> Callable:
        # Should do nothing
        def collate(batch, destination):
            destination[:] = batch
            return destination
        return collate
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.stage == Stage.INDIVIDUAL
        return replace(previous_state, stage=Stage.BATCH), None

class ToTensor(CoreOp):
    def __init__(self):
        super().__init__()
    
    def generate_code(self) -> Callable:
        def to_tensor(image, dst, _):
            return ch.from_numpy(image)
        return to_tensor
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.stage == Stage.BATCH
        return previous_state, AllocationQuery((3, 32, 32), dtype=np.dtype('uint8'))
