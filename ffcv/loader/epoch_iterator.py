import ast
from time import sleep, time
from collections import defaultdict
from itertools import zip_longest
from functools import partial
from threading import Thread
from queue import Queue
from typing import Sequence, TYPE_CHECKING, Mapping

import numpy as np

from ffcv.pipeline.compiler import Compiler

from ..utils import chunks

if TYPE_CHECKING:
    from .loader import Loader
    from ..pipeline.pipeline import Pipeline
    
class EpochIterator(Thread):

    def __init__(self, loader: 'Loader', epoch: int, order:Sequence[int]):
        super().__init__(daemon=True)
        self.loader: 'Loader' = loader
        self.order = order
        self.idx_iter = iter(order)
        self.storage_state = loader.memory_manager.state
        self.metadata = loader.reader.metadata
        self.current_batch_slot = 0
        self.epoch = epoch
        self.iter_ixes = iter(chunks(order, self.loader.batch_size))

        self.memory_bank_per_stage = defaultdict(list)

        # Allocate all the memory
        memory_allocations = {} 
        for (p_id, p) in self.loader.pipelines.items():
            memory_allocations[p_id] = p.allocate_memory(self.loader.batch_size,
                                                         self.loader.batches_ahead + 2)
        
        # Assign each memory bank to the pipeline stage it belongs to
        for s_ix, banks in self.loader.memory_bank_keys_per_stage.items():
            for (pipeline_name, op_id) in banks:
                self.memory_bank_per_stage[s_ix].append(
                        memory_allocations[pipeline_name][op_id]
                )
                
        self.output_queue = Queue(self.loader.batches_ahead)
                
        self.start()

    def before_epoch(self):
        if self.code_per_stage is None:
            self.generate_code()
            
    def run(self):
        try:
            while True:
                ixes = next(self.iter_ixes)
                slot = self.current_batch_slot
                self.current_batch_slot = (slot + 1) % (self.loader.batches_ahead + 2)
                result = self.run_pipeline(ixes, slot)
                self.output_queue.put(result)
        except StopIteration:
            self.output_queue.put(None)
            

    def run_pipeline(self, batch_indices, batch_slot):
        args = [batch_indices]
        for stage, banks in self.memory_bank_per_stage.items():
            for bank in banks:
                if bank is not None:
                    bank = bank[batch_slot]
                args.append(bank)
            args.append(self.metadata)
            args.append(self.storage_state)
            code = self.loader.code_per_stage[stage]
            result = code(*args)
            args = list(result)
        return tuple(args)

        
    def __next__(self):
        result = self.output_queue.get()
        # print("EXTRACTED", result[0], result[1][0])
        if result is None:
            raise StopIteration()
        return result
