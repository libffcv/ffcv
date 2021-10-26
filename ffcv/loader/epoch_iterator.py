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
        self.before_epoch()
        self.generated_code = self.generate_code()
        self.current_batch_slot = 0
        self.epoch = epoch
        self.iter_ixes = iter(chunks(order, self.loader.batch_size))

    def before_epoch(self):
        for name in self.loader.reader.handlers:
            self.loader.pipelines[name].before_epoch(self.loader.batch_size,
                                                        self.batches_ahead)

    def generate_function_call(self, p_ix, pipeline_name, op_id):
        pipeline_identifier = f'code_{pipeline_name}_{op_id}'
        memory_identifier = f'memory_{pipeline_name}_{op_id}'
        result_identifier = f'result_{pipeline_name}'

        tree = ast.parse(f"""
{result_identifier} = {pipeline_identifier}({result_identifier}, {memory_identifier})
        """).body[0]
        return tree

    def generate_stage_code(self, stage, stage_ix):
        base_code = ast.parse(f"""
def stage_{stage_ix}(toto):
    pass
        """).body[0]

        function_calls = []
        memory_banks = []
        for p_ix, pipeline_name, op_id in stage:
            function_calls.append(self.generate_function_call(p_ix,
                                                              pipeline_name,
                                                              op_id))
            arg = ast.arg(arg=f'memory_{pipeline_name}_{op_id}')
            memory_banks.append(arg)


        base_code.body.pop()
        base_code.body.extend(function_calls)

        return_tuple = ast.Return(value=ast.Tuple(elts=[], ctx=ast.Load()))

        for p_id in self.loader.pipelines.keys():
            r = f'result_{p_id}'
            base_code.args.args.append(ast.arg(arg=r))
            return_tuple.value.elts.append(ast.Name(id=r, ctx=ast.Load()))
        base_code.body.append(return_tuple)
        base_code.args.args.extend(memory_banks)

        return base_code

    def generate_code(self):
        stages = []
        pipelines: Mapping[str: 'Pipeline'] =  self.loader.pipelines

        pipeline_keys = list(pipelines.keys())

        schedule = defaultdict(lambda: [])
        for p_ix, (p_id, p) in enumerate(self.loader.pipelines.items()):
            stage = 0
            for jitted_block, block_content in p.operation_blocks:
                # Even stages are jitted Odds are not
                # If this doesn't match for this pipeline we
                # shift the operations
                if 1 - jitted_block % 2 != stage % 2:
                    stage += 1
                for op in block_content:
                    schedule[stage].append((p_ix, p_id, op))
                stage += 1

        for stage_ix, stage in schedule.items():
            code_for_stage = self.generate_stage_code(stage, stage_ix)
        return


                    
        
    def __next__(self):
        ixes = next(self.iter_ixes)
        slot = self.current_batch_slot
        result = self.generated_code(slot, ixes)
        self.current_batch_slot = (slot + 1) % self.batches_ahead
        return result
