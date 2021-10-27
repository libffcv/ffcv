import ast
from collections import defaultdict
from itertools import zip_longest
from functools import partial
from threading import Thread
from typing import Sequence, TYPE_CHECKING, Mapping

import numpy as np

from ffcv.pipeline.compiler import Compiler

from ..utils import chunks

if TYPE_CHECKING:
    from .loader import Loader
    from ..pipeline.pipeline import Pipeline

class EpochIterator(Thread):

    # TODO REUSE Iterators multiple time
    def __init__(self, loader: 'Loader', epoch: int, order:Sequence[int]):
        self.loader: 'Loader' = loader
        self.order = order
        self.idx_iter = iter(order)
        self.batches_ahead = 3
        self.code_per_stage = None
        self.memory_bank_per_stage = {}
        self.before_epoch()
        self.current_batch_slot = 0
        self.epoch = epoch
        self.iter_ixes = iter(chunks(order, self.loader.batch_size))

    def before_epoch(self):
        for name in self.loader.reader.handlers:
            self.loader.pipelines[name].before_epoch(self.loader.batch_size,
                                                        self.batches_ahead)
            
        if self.code_per_stage is None:
            self.generate_code()
            
        self.memory_bank_per_stage = defaultdict(list)

        for s_ix, banks in self.memory_bank_keys_per_stage.items():
            for (pipeline_name, op_id) in banks:
                self.memory_bank_per_stage[s_ix].append(
                    self.loader.pipelines[pipeline_name].memory_buffers[op_id])


    def generate_function_call(self, p_ix, pipeline_name, op_id):
        pipeline_identifier = f'code_{pipeline_name}_{op_id}'
        memory_identifier = f'memory_{pipeline_name}_{op_id}'
        result_identifier = f'result_{pipeline_name}'
        
        arg_id = result_identifier
        # This is the decoder so we pass the indices instead of the previous
        # result
        if op_id == 0:
            arg_id = 'batch_indices'

        tree = ast.parse(f"""
{result_identifier} = {pipeline_identifier}({arg_id}, {memory_identifier})
        """).body[0]
        return tree

    def generate_stage_code(self, stage, stage_ix, functions):
        fun_name = f'stage_{stage_ix}'
        base_code = ast.parse(f"""
def {fun_name}():
    pass
        """).body[0]

        function_calls = []
        memory_banks = []
        memory_banks_id = []
        for p_ix, pipeline_name, op_id in stage:
            function_calls.append(self.generate_function_call(p_ix,
                                                              pipeline_name,
                                                              op_id))
            arg = ast.arg(arg=f'memory_{pipeline_name}_{op_id}')
            memory_banks.append(arg)
            memory_banks_id.append((pipeline_name, op_id))


        base_code.body.pop()
        base_code.body.extend(function_calls)

        return_tuple = ast.Return(value=ast.Tuple(elts=[], ctx=ast.Load()))

        for p_id in self.loader.pipelines.keys():
            r = f'result_{p_id}'
            if stage_ix != 0:
                base_code.args.args.append(ast.arg(arg=r))
            return_tuple.value.elts.append(ast.Name(id=r, ctx=ast.Load()))

        if stage_ix == 0:
            base_code.args.args.append(ast.arg(arg='batch_indices'))

        base_code.body.append(return_tuple)
        base_code.args.args.extend(memory_banks)
        
        module = ast.fix_missing_locations(
            ast.Module(body=[base_code],
                       type_ignores=[])
        )
        namespace = {
            **functions,
        }

        exec(compile(module, '', 'exec'), namespace) 
        final_code = namespace[fun_name]
        if stage_ix % 2 == 0:
            final_code = Compiler.compile(final_code)

        return final_code, memory_banks_id

    def generate_code(self):
        schedule = defaultdict(lambda: [])
        compiled_functions = {}
        for p_ix, (p_id, p) in enumerate(self.loader.pipelines.items()):
            stage = 0
            for jitted_block, block_content in p.operation_blocks:
                # Even stages are jitted Odds are not
                # If this doesn't match for this pipeline we
                # shift the operations
                if 1 - jitted_block % 2 != stage % 2:
                    stage += 1
                for op in block_content:
                    ops_code = p.compiled_ops[op]
                    if stage % 2 == 0:
                        ops_code = Compiler.compile(ops_code)
                    compiled_functions[f'code_{p_id}_{op}'] = ops_code
                    schedule[stage].append((p_ix, p_id, op))
                stage += 1
                
        memory_bank_keys_per_stage = {}
        self.code_per_stage = {}
        for stage_ix, stage in schedule.items():
            code_for_stage, mem_banks_ids = self.generate_stage_code(stage, stage_ix,
                                                                     compiled_functions)
            if stage_ix % 2 == 0:
                code_for_stage = Compiler.compile(code_for_stage)
            self.code_per_stage[stage_ix] = code_for_stage
            memory_bank_keys_per_stage[stage_ix] = mem_banks_ids
        
        self.memory_bank_keys_per_stage = memory_bank_keys_per_stage
        
    def run_pipeline(self, batch_indices, batch_slot):
        args = [batch_indices]
        for stage, banks in self.memory_bank_per_stage.items():
            for bank in banks:
                if bank is not None:
                    bank = bank[batch_slot ]
                args.append(bank)
            code = self.code_per_stage[stage]
            result = code(*args)
            args = list(result)
        return tuple(args)

        
    def __next__(self):
        ixes = next(self.iter_ixes)
        slot = self.current_batch_slot
        self.current_batch_slot = (slot + 1) % self.batches_ahead
        return self.run_pipeline(ixes, slot)
