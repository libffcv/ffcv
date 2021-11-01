import ast
from dataclasses import replace
from typing import Any, List, Optional, Sequence, Mapping, Tuple

import torch as ch
import numpy as np

from .state import State
from .compiler import Compiler
from .operation import Operation
from .allocation_query import AllocationQuery
from ..transforms.ops import Collate, ToTensor, CoreOp
from ..transforms.module import ModuleWrapper

BAD_COLLATION_MESSAGE: str = "Each pipeline needs one and one only Collate operation"


class Pipeline:

    def __init__(self, operations: Sequence[Operation]):

        # This is the starting state of the pipeline
        self.original_state = State(jit_mode=True,
                                    device=ch.device('cpu'),
                                    dtype=np.dtype('u1'),
                                    shape=None)

        self.operations = operations

        self.operation_blocks, _ = self.parse_pipeline()
        self.compiled_ops = self.compile_ops()

        # Compile the pipeline
        self.compiled_code = None

    def parse_pipeline(self, batch_size=16):
        memory_allocations: Mapping[int, Optional[AllocationQuery]] = {}
        operation_blocs = []

        current_state: State = self.original_state
        current_block = []

        # We read the content of the pipeline, validate and collect
        # Memory allocations
        for op_id, operation in enumerate(self.operations):
            previous_state = current_state
            current_state, memory_allocation = operation.declare_state_and_memory(
                current_state)

            if current_state.jit_mode != previous_state.jit_mode:
                if current_block:
                    operation_blocs.append((previous_state.jit_mode, current_block))
                current_block = [op_id]
            else:
                current_block.append(op_id)

            memory_allocations[op_id] = memory_allocation

        if current_block:
            operation_blocs.append((current_state.jit_mode, current_block))

        return operation_blocs, memory_allocations
        
    def compile_ops(self):
        compiled_ops = {}
        for op_id, operation in enumerate(self.operations):
            compiled_ops[op_id] = operation.generate_code()
        return compiled_ops
        

    def allocate_memory(self, batch_size: int, batches_ahead: int):
        _, memory_allocations = self.parse_pipeline()

        # Contains the actual allocated memory
        memory_buffers: Mapping[int, Any] = {}

        # For each allocation made by the operations in the pipeline
        for op_id, memory_allocation in memory_allocations.items():
            # If the operation didn't make a query we stop here
            if memory_allocation is None:
                memory_buffers[op_id] = None
            else:
                # We compute the total amount of memory needed for this
                # operation
                final_shape = [batches_ahead,
                               batch_size, *memory_allocation.shape]
                if isinstance(memory_allocation.dtype, ch.dtype):
                    result = ch.empty(*final_shape,
                                      dtype=memory_allocation.dtype,
                                      device=memory_allocation.device)
                    try:
                        result = result.pin_memory()
                    except:
                        pass
                else:
                    ch_dtype = ch.from_numpy(np.empty(0, dtype=memory_allocation.dtype)).dtype
                    result = ch.empty(*final_shape,
                                      dtype=ch_dtype)
                    try:
                        result = result.pin_memory()
                    except:
                        pass
                    result = result.numpy()

                memory_buffers[op_id] = result

        return memory_buffers
