import ast
from typing import Sequence, Mapping

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
        
        current_state = self.original_state
        self.operations = operations
        
        # Contains the actual allocated memory
        self.memory_buffers = []
        # Where we remember what each operation in the pipeline needs
        self.memory_allocations : Mapping[int, AllocationQuery] = {}

        # We read the content of the pipeline, validate and collect
        # Memory allocations
        for op_id, operation in enumerate(operations):
            current_state, memory_allocation = operation.declare_state_and_memory(current_state)
            self.memory_allocations[op_id] = memory_allocation
            
        # Compile the pipeline
        self.compiled_code = None
            
    def allocate_memory(self, batch_size: int, batches_ahead: int):
        for op_id, memory_allocation in self.memory_allocations.items():
            if memory_allocation is None:
                result = None
            else:
                final_shape = [batches_ahead,
                               batch_size, *memory_allocation.shape]
                print(op_id, final_shape)
                result = np.empty(final_shape,
                                  dtype=memory_allocation.dtype)
            self.memory_buffers.append(result)
        
    def run(self):
        pass
        
    def generate_code(self, memory):
        
        # BIG METAPROGRAMMING PARTY INCOMING
        
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
        
