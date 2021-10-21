from typing import Sequence, Mapping

import torch as ch
import numpy as np

from .state import State
from .stage import Stage
from .operation import Operation
from .allocation_query import AllocationQuery

class Pipeline:

    def __init__(self, operations: Sequence[Operation]):
        
        # This is the starting state of the pipeline
        self.original_state = State(stage=Stage.INDIVIDUAL,
                                    jit_mode=True,
                                    device=ch.device('cpu'),
                                    shape=None)
        
        current_state = self.original_state
        self.operations = operations
        
        # Contains the actual allocated memory
        self.memory_buffers = {}
        # Where we remember what each operation in the pipeline needs
        self.memory_allocations : Mapping[int, AllocationQuery] = {}

        # We read the content of the pipeline, validate and collect
        # Memory allocations
        for op_id, operation in enumerate(operations):
            current_state, memory_allocation = operation.declare_state_and_memory(current_state)
            self.memory_allocations[op_id] = memory_allocation
            
    def allocate_memory(self, batch_size):
        for op_id, memory_allocation in self.memory_allocations.items():
            if memory_allocation is None:
                result = None
            else:
             result = np.empty(memory_allocation.shape,
                               dtype=memory_allocation.dtype)
            self.memory_buffers[op_id] = result
        
    def run(self):
        pass
        
    def compile(self):
        pass
