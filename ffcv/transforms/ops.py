"""
General operations:

- Collation
- Conversion to PyTorch Tensor
"""
from abc import ABCMeta
import torch as ch
import numpy as np
from typing import Callable, Optional, Tuple
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.stage import Stage
from ..pipeline.state import State
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
        def to_tensor(inp, dst):
            return ch.from_numpy(inp)
        return to_tensor
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        new_dtype = ch.from_numpy(np.empty((), dtype=previous_state.dtype)).dtype
        assert previous_state.stage == Stage.BATCH
        # TODO: is this the right place to turn off jit?
        return replace(previous_state, stage=Stage.PYTORCH, jit_mode=False, dtype=new_dtype), None

class ToDevice(CoreOp):
    def __init__(self, device, non_blocking=True):
        super().__init__()
        self.device = device
        self.non_blocking = non_blocking
    
    def generate_code(self) -> Callable:
        def to_device(inp, dst):
            dst[:inp.shape[0]].copy_(inp, non_blocking=self.non_blocking)
            return dst[:inp.shape[0]]

        return to_device

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.stage == Stage.PYTORCH
        return replace(previous_state, device=self.device), AllocationQuery(previous_state.shape, dtype=previous_state.dtype, device=self.device)

class ToTorchImage(CoreOp):
    def __init__(self, channels_last=True):
        super().__init__()
        self.channels_last = channels_last
    
    def generate_code(self) -> Callable:
        def to_torch_image(inp: ch.Tensor, dst):
            # Returns a permuted view of the same tensor
            inp = inp.permute([0, 3, 1, 2]) 
            # If channels last, it's already contiguous so we're good
            if self.channels_last:
                assert inp.is_contiguous(memory_format=ch.channels_last)
                return inp

            # Otherwise, need to fill the allocated memory with the contiguous tensor
            dst[:inp.shape[0]] = inp.contiguous()
            return dst[:inp.shape[0]]
        
        return to_torch_image
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        alloc = None
        H, W, C = previous_state.shape
        if not self.channels_last:
            alloc = AllocationQuery((C, H, W), dtype=previous_state.dtype)
        return replace(previous_state, shape=(C, H, W)), alloc
    
class Convert(CoreOp):
    def __init__(self, target_dtype):
        super().__init__()
        self.target_dtype = target_dtype
    
    def generate_code(self) -> Callable:
        def convert(inp, dst):
            return inp.type(self.target_dtype)

        return convert
    
    # TODO: something weird about device to allocate on
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return replace(previous_state, dtype=self.target_dtype), None