import ast
from collections import defaultdict
from dataclasses import replace
from typing import List, Optional, Sequence, Mapping

import torch as ch
import numpy as np

from .state import State
from .compiler import Compiler
from .stage import Stage, ALL_STAGES
from .operation import Operation
from .allocation_query import AllocationQuery
from ..transforms.ops import Collate, ToTensor, CoreOp

BAD_COLLATION_MESSAGE: str = "Each pipeline needs one and one only Collate operation"

class Pipeline:

    def __init__(self, operations: Sequence[Operation]):
        
        # This is the starting state of the pipeline
        self.original_state = State(stage=Stage.INDIVIDUAL,
                                    jit_mode=True,
                                    device=ch.device('cpu'),
                                    dtype=np.dtype('u1'),
                                    shape=None)
        
        self.operations = operations
        self.per_stage_operations: Mapping[ALL_STAGES, List[Operation]] = self.parse_pipeline()[0]
        
        print(self.per_stage_operations)
        
        # Contains the actual allocated memory
        self.memory_buffers: Mapping[int, np.ndarray] = {}
        # Where we remember what each operation in the pipeline needs

        # Compile the pipeline
        self.compiled_code = None
        
    def parse_pipeline(self, batch_size=16):
        memory_allocations : Mapping[int, Optional[AllocationQuery]] = {}
        current_state: State = self.original_state

        per_stage_operations: Mapping[ALL_STAGES, List[Operation]] = defaultdict(list)
        
        has_collate: bool = False
        

        # We read the content of the pipeline, validate and collect
        # Memory allocations
        for op_id, operation in enumerate(self.operations):
            previous_state = current_state
            current_state, memory_allocation = operation.declare_state_and_memory(current_state)
            memory_allocations[op_id] = memory_allocation
            
            # Check that the operation is not changing the pipeline
            # When it should not
            if (not isinstance(operation, CoreOp)
                    and current_state.stage != previous_state.stage):
                raise AssertionError("You are not allowed to change the stage")
            
            # Add batch size to the shape when collating
            if isinstance(operation, Collate):
                # We can't have a second
                if has_collate:
                    raise ValueError(BAD_COLLATION_MESSAGE)
                has_collate = True
                final_shape = (batch_size,) + current_state.shape
                # We allocate memory for the collated data
                memory_allocations[op_id] = AllocationQuery(final_shape,current_state.dtype)
                current_state = replace(current_state,
                                        shape=final_shape)

            per_stage_operations[previous_state.stage].append(operation)

        if not has_collate:
            raise ValueError(BAD_COLLATION_MESSAGE)
            
        return per_stage_operations, memory_allocations
        
    def before_epoch(self, batch_size: int, batches_ahead: int):
        
        _, memory_allocations = self.parse_pipeline()
        print(memory_allocations)
            
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
                    if (current_buffer.shape == final_shape and
                            current_buffer.dtype == memory_allocation.dtype):
                        result = self.memory_buffers[op_id]
                if result is None:
                    result = np.empty(final_shape,
                                      dtype=memory_allocation.dtype)
                self.memory_buffers[op_id] = result

    def generate_code(self, stage:ALL_STAGES):
        
        # TODO do not recompile multiple times
        if stage == Stage.INDIVIDUAL:
            return self.generate_code_for_samples()
        if stage==Stage.BATCH:
            return self.generate_code_for_batches()

        
                
    def generate_code_for_samples(self):
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
                                                        ast.Name( id=dst_name, ctx=ast.Load())
                                                        ])))
        body.append(ast.Return(value=ast.Name(id='result', ctx=ast.Load())))
            
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
        }
        
        from astor import to_source
        # print(to_source(module))
        
        exec(compile(module, '', 'exec'), namespace)
        return Compiler.compile(namespace['compute_pipeline'])
    
