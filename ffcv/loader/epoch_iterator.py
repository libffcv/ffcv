from threading import Thread
from typing import Sequence, TYPE_CHECKING

from ..utils import chunks

if TYPE_CHECKING:
    from .loader import Loader

class EpochIterator(Thread):

    def __init__(self, loader: 'Loader', epoch: int, order:Sequence[int]):
        self.loader: 'loader' = loader
        self.order = order
        self.idx_iter = iter(order)
        self.batches_ahead = 3
        self.allocate_memory()
        self.generated_code = self.generate_code()
        self.current_batch_slot = 0
        self.iter_ixes = iter(chunks(order, self.loader.batch_size))
        
    def allocate_memory(self):
        for name in self.loader.reader.handlers:
            self.loader.pipelines[name].allocate_memory(self.loader.batch_size,
                                                        self.batches_ahead)
            
    def generate_code(self):
        pipelines = []
        memories = []

        for name in self.loader.reader.handlers:
            pipeline = self.loader.pipelines[name]
            pipelines.append(pipeline.generate_code(self.loader.memory_manager))
            memories.append(pipeline.memory_buffers)
            
            
        metadata = self.loader.reader.metadata

        def compute(batch_slot, batch_indices):
            for dest_ix, ix in enumerate(batch_indices):
                sample = metadata[ix]
                for p_ix in range(len(pipelines)):
                    field_value = sample[p_ix]
                    mem_banks = []
                    for mem in memories[p_ix]:
                        mem_banks.append(mem[batch_slot, dest_ix])
                    pipelines[p_ix](field_value, *mem_banks)
                    
            return memories
        return compute

                    
        
    def __next__(self):
        ixes = next(self.iter_ixes)
        slot = self.current_batch_slot
        result = self.generated_code(slot,
                                     ixes)
        self.current_batch_slot = (slot + 1) % self.batches_ahead
        return [
            x[-1][slot]
            for x in result
        ]
