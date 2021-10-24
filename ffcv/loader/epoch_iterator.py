import ast
from threading import Thread
from typing import Sequence, TYPE_CHECKING

import numpy as np

from ffcv.pipeline.compiler import Compiler

from ..utils import chunks
from ..pipeline.state import Stage

if TYPE_CHECKING:
    from .loader import Loader

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
            
    def generate_code(self):
        pipelines_sample = []
        pipelines_batch = []
        pipelines_pytorch = []
        memories_sample = []
        memories_batch = []
        memories_pytorch = []

        # TODO stop copy/paste please G.
        for name in self.loader.reader.handlers:
            pipeline = self.loader.pipelines[name]
            pipelines_sample.append(pipeline.generate_code(Stage.INDIVIDUAL))
            pipelines_batch.append(pipeline.generate_code(Stage.BATCH))
            pipelines_pytorch.append(pipeline.generate_code(Stage.PYTORCH))
            memories_sample.append(pipeline.memory_for_stage(Stage.INDIVIDUAL))
            memories_batch.append(pipeline.memory_for_stage(Stage.BATCH))
            memories_pytorch.append(pipeline.memory_for_stage(Stage.PYTORCH))


        metadata = self.loader.reader.metadata

        import astor
        function_calls = []
        per_sample_namespace = {
            'my_range': Compiler.get_iterator()
        }
        return_values = []
        for p_ix in range(len(pipelines_sample)):
            mem_bank_exprs = []
            pipeline_identifier = f"pipeline_{p_ix}_code"
            per_sample_namespace[pipeline_identifier] = pipelines_sample[p_ix]
            for mem_id, mem in enumerate(memories_sample[p_ix]):
                mem_identifier = f"mem_bank_p_{p_ix}_id_{mem_id}"
                per_sample_namespace[mem_identifier] = mem
                if mem is None:
                    mem_bank_exprs.append(
                        ast.Constant(value=None)
                    )
                else:
                    mem_bank_exprs.append(
                        ast.parse(f"{mem_identifier}[batch_slot, dest_ix]")
                    )
            return_values.append(ast.Name(id=mem_identifier, ctx=ast.Load()))
            f_call = ast.Call(func=ast.Name(id=pipeline_identifier, ctx=ast.Load()),
                              keywords=[],
                              args=[ast.Name(id='field_value', ctx=ast.Load()), *mem_bank_exprs])
            f_call = ast.Expr(value=f_call, decorator_list=[], lineno=p_ix+2)
            function_calls.append(f_call)

        base_code = ast.parse("""
def compute_samplewise(batch_slot, batch_indices):
    for ix in my_range(len(batch_indices)):
        dest_ix = batch_indices[ix]
        sample = metadata[ix]
        """)
        # Append the content of the pipeline to the for loop
        base_code.body[0].body[0].body.extend(function_calls)

        base_code.body[0].body.append(ast.Return(value=ast.Tuple(elts=return_values,
                                                                 ctx=ast.Load())))
        base_code = ast.fix_missing_locations(base_code)

        # Add the proper return statement
        print(astor.to_source(base_code))

        exec(compile(base_code, '', 'exec'), per_sample_namespace)


        def compute_sample(batch_slot, batch_indices):
            # For each sample
            for dest_ix, ix in enumerate(batch_indices):
                sample = metadata[ix]
                # For each field/pipline
                for p_ix in range(len(pipelines_sample)):
                    field_value = sample[p_ix]
                    memory_banks = []
                    for mem in memories_sample[p_ix]:
                        if mem is None:
                            memory_banks.append(None)
                        else:
                            memory_banks.append(mem[batch_slot, dest_ix])

                    np.random.seed(int(self.loader.reader.num_samples * self.epoch + ix + self.loader.seed))
                    pipelines_sample[p_ix](field_value, *memory_banks)

            final_result = []

            # Because we know that the last per-sample operation is Collate()
            # which allocates its own memory, we can just look in the final
            # allocated memory slot for the final result

            # TODO select the proper subset of the instead of the whole thing!
            for res in memories_sample:
                final_result.append(res[-1][batch_slot, :len(batch_indices)])
            return final_result
            
        def compute_batch(batch_slot, batch_indices):
            batches = compute_sample(batch_slot, batch_indices)
            result = []
            for batch, op, mems in zip(batches, pipelines_batch, memories_batch):
                result.append(op(batch, *[None if mem is None else mem[batch_slot] for mem in mems]))

            return tuple(result)

        def compute_pytorch(batch_slot, batch_indices):
            batches = compute_batch(batch_slot, batch_indices)
            result = []
            for batch, op, mems in zip(batches, pipelines_pytorch, memories_pytorch):
                result.append(op(batch, *[None if mem is None else mem[batch_slot] for mem in mems]))

            return tuple(result)
            
        return compute_pytorch

                    
        
    def __next__(self):
        ixes = next(self.iter_ixes)
        slot = self.current_batch_slot
        result = self.generated_code(slot, ixes)
        self.current_batch_slot = (slot + 1) % self.batches_ahead
        return result
