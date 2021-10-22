from threading import Thread
from typing import Sequence, TYPE_CHECKING

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
                    pipelines_sample[p_ix](field_value, *memory_banks)
                    
            final_result = []
            for res in memories_sample:
                final_result.append(res[-1][batch_slot, :len(batch_indices)])
            return final_result
            
        def compute_batch(batch_slot, batch_indices):
            batches = compute_sample(batch_slot, batch_indices)
            
            result = []
            for batch, op, mems in zip(batches, pipelines_batch, memories_batch):
                result.append(op(batch, *mems))

            return tuple(result)

        def compute_pytorch(batch_slot, batch_indices):
            batches = compute_batch(batch_slot, batch_indices)
            
            result = []
            for batch, op, mems in zip(batches, pipelines_pytorch, memories_pytorch):
                result.append(op(batch, *mems))

            return tuple(result)
            
        return compute_pytorch

                    
        
    def __next__(self):
        ixes = next(self.iter_ixes)
        slot = self.current_batch_slot
        result = self.generated_code(slot, ixes)
        self.current_batch_slot = (slot + 1) % self.batches_ahead
        return result
