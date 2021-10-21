import ast
from typing import List, Optional, Sequence, Mapping

import torch as ch
import numpy as np

from .state import State
from .compiler import Compiler
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
        
        self.operations = operations
        
        # Contains the actual allocated memory
        self.memory_buffers: Mapping[int, np.ndarray] = {}
        # Where we remember what each operation in the pipeline needs

        # Compile the pipeline
        self.compiled_code = None
        
    def before_epoch(self, batch_size: int, batches_ahead: int):

        memory_allocations : Mapping[int, Optional[AllocationQuery]] = {}
        current_state = self.original_state

        # We read the content of the pipeline, validate and collect
        # Memory allocations
        for op_id, operation in enumerate(self.operations):
            current_state, memory_allocation = operation.declare_state_and_memory(current_state)
            memory_allocations[op_id] = memory_allocation

            
        for op_id, memory_allocation in memory_allocations.items():
            if memory_allocation is not None:
                final_shape = [batches_ahead,
                               batch_size, *memory_allocation.shape]
                result = None
                # We only allocate memory if:
                # - it wansn't previously or
                # - the new request is different than what was allocated
                if op_id in self.memory_buffers:
                    current_buffer = self.memory_buffers[op_id]
                    if (current_buffer.shape == final_shape and
                            current_buffer.dtype == memory_allocation.dtype):
                        result = self.memory_buffers[op_id]
                if result is None:
                    result = np.empty(final_shape,
                                      dtype=memory_allocation.dtype)
                self.memory_buffers[op_id] = result
        
    def generate_code(self, memory):
        
        # BIG METAPROGRAMMING PARTY INCOMING
        
        # TODO do not recompile multiple times

        arguments = [ast.arg('result')]
        body = []
        functions = {}

        for op_id, op in enumerate(self.operations):
            func_name = f'pipeline_stage_{op_id}'
            functions[func_name] = Compiler.compile(op.generate_code())
            dst_name = f"dest_{op_id}"
            arguments.append(ast.arg(dst_name))
            body.append(ast.Assign(targets=[ast.Name(id='result', ctx=ast.Store())],
                                   value=ast.Call(func=ast.Name(id=func_name, ctx=ast.Load()),
                                                  keywords=[],
                                                  args=[ast.Name('result', ctx=ast.Load()),
                                                        ast.Name( id=dst_name, ctx=ast.Load()),
                                                        ast.Name( id='memory', ctx=ast.Load())
                                                        ])))
            
        final_function = ast.FunctionDef(name='compute_pipeline',
                                         args=ast.arguments(args=arguments,
                                                            posonlyargs=[],
                                                            kwonlyargs=[],
                                                            kw_defaults=[],
                                                            defaults=[]),
                                         body=body,
                                         decorator_list=[])
        
        module = ast.fix_missing_locations(
            ast.Module(body=[final_function],
                            type_ignores=[])
        )
        
        namespace = {
            **functions,
            'memory': memory
        }
        
        exec(compile(module, '', 'exec'), namespace)
        return Compiler.compile(namespace['compute_pipeline'])
        
