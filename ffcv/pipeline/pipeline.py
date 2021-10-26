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
        for i, op in enumerate(self.operations):
            if isinstance(op, ch.nn.Module):
                self.operations[i] = ModuleWrapper(op)

        # Contains the actual allocated memory
        self.memory_buffers: Mapping[int, Any] = {}
        # Where we remember what each operation in the pipeline needs

        self.operation_blocks, _ = self.parse_pipeline()

        print(self.operation_blocks)

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

    def before_epoch(self, batch_size: int, batches_ahead: int):
        _, memory_allocations = self.parse_pipeline()

        for op_id, memory_allocation in memory_allocations.items():
            if memory_allocation is None:
                self.memory_buffers[op_id] = None
            else:
                final_shape = [batches_ahead,
                               batch_size, *memory_allocation.shape]
                result = None
                # We only allocate memory if:
                # - it wansn't previously or
                # - the new request is different than what was allocated
                if op_id in self.memory_buffers:
                    current_buffer = self.memory_buffers[op_id]
                    is_pytorch = isinstance(current_buffer, ch.Tensor)

                    # Check to make sure the buffer fits the bill
                    shape_matches = current_buffer.shape == final_shape
                    type_matches = is_pytorch == isinstance(
                        memory_allocation.dtype, ch.dtype)
                    dtype_matches = current_buffer.dtype == memory_allocation.dtype
                    device_matches = (not is_pytorch) or (
                        current_buffer.device == memory_allocation.device)

                    if (not isinstance(memory_allocation.dtype, ch.dtype)) and (memory_allocation.device is not None):
                        raise ValueError(
                            'Numpy allocations must be made on CPU.')

                    if shape_matches and dtype_matches and type_matches and device_matches:
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