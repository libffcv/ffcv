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

        # Contains the actual allocated memory
        self.memory_buffers: Mapping[int, Any] = {}
        # Where we remember what each operation in the pipeline needs

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
        

    def before_epoch(self, batch_size: int, batches_ahead: int):
        _, memory_allocations = self.parse_pipeline()

        # For each allocation made by the operations in the pipeline
        for op_id, memory_allocation in memory_allocations.items():
            # If the operation didn't make a query we stop here
            if memory_allocation is None:
                self.memory_buffers[op_id] = None
            else:
                # We compute the total amount of memory needed for this
                # operation
                final_shape = [batches_ahead,
                               batch_size, *memory_allocation.shape]
                result = None

                # We try to reuse previously allocated memory
                # - it wansn't previously or
                # - the new request is different than what was allocated
                if op_id in self.memory_buffers:
                    # => There already was a previously allocated
                    current_buffer = self.memory_buffers[op_id]
                    is_pytorch = isinstance(current_buffer, ch.Tensor)

                    # Check to make sure the buffer fits the bill
                    shape_matches = current_buffer.shape == final_shape
                    dtype_matches = current_buffer.dtype == memory_allocation.dtype
                    device_matchs = True
                    if is_pytorch:
                        device_matchs = (current_buffer.device
                                         == memory_allocation.device)
                        
                    if shape_matches and dtype_matches and device_matchs:
                        result = self.memory_buffers[op_id]
                if result is None:
                    if isinstance(memory_allocation.dtype, ch.dtype):
                        result = ch.empty(*final_shape,
                                          dtype=memory_allocation.dtype,
                                          device=memory_allocation.device)
                    else:
                        result = np.empty(final_shape,
                                          dtype=memory_allocation.dtype)

                self.memory_buffers[op_id] = result
