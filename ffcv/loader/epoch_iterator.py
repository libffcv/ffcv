from collections import defaultdict
from threading import Thread, Event
from queue import Queue, Full
from contextlib import nullcontext
from typing import Sequence, TYPE_CHECKING

import torch as ch

from ..traversal_order.quasi_random import QuasiRandom
from ..utils import chunks
from ..pipeline.compiler import Compiler

if TYPE_CHECKING:
    from .loader import Loader
    
IS_CUDA = ch.cuda.is_available()

QUASIRANDOM_ERROR_MSG = '''Not enough memory; try setting quasi-random ordering
(`OrderOption.QUASI_RANDOM`) in the dataloader constructor's `order` argument.
'''

class EpochIterator(Thread):
    def __init__(self, loader: 'Loader', order: Sequence[int]):
        super().__init__(daemon=True)
        self.loader: 'Loader' = loader
        self.order = order
        self.metadata = loader.reader.metadata
        self.current_batch_slot = 0
        batches = list(chunks(order, self.loader.batch_size))
        self.iter_ixes = iter(batches)
        self.closed = False
        self.output_queue = Queue(self.loader.batches_ahead)
        self.terminate_event = Event()
        self.memory_context = self.loader.memory_manager.schedule_epoch(
            batches)
        try:
            self.memory_context.__enter__()
        except MemoryError as e:
            if loader.traversal_order != QuasiRandom:
                print(QUASIRANDOM_ERROR_MSG)
                print('Full error below:')

            raise e

        self.storage_state = self.memory_context.state

        self.memory_bank_per_stage = defaultdict(list)

        self.cuda_streams = [(ch.cuda.Stream() if IS_CUDA else None)
                             for _ in range(self.loader.batches_ahead + 2)]

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

        self.start()

    def run(self):

        events = [None for _ in self.cuda_streams]

        try:
            b_ix = 0
            Compiler.set_num_threads(self.loader.num_workers)
            while True:
                ixes = next(self.iter_ixes)
                slot = self.current_batch_slot
                self.current_batch_slot = (
                    slot + 1) % (self.loader.batches_ahead + 2)
                result = self.run_pipeline(b_ix, ixes, slot, events[slot])
                to_output = (slot, result)
                while True:
                    try:
                        self.output_queue.put(to_output, block=True, timeout=0.5)
                        break
                    except Full:
                        pass

                    if self.terminate_event.is_set():
                        return
                if IS_CUDA:
                    # We were able to submit this batch
                    # Therefore it means that the user must have entered the for loop for
                    # (batch_slot - batch_ahead + 1) % (batches ahead + 2)
                    # Therefore batch_slot - batch_ahead must have all it's work submitted
                    # We will record an event of all the work submitted on the main stream
                    # and make sure no one overwrite the data until they are done
                    just_finished_slot = (slot - self.loader.batches_ahead) % (self.loader.batches_ahead + 2)
                    event = ch.cuda.Event()
                    event.record(ch.cuda.default_stream())
                    events[just_finished_slot] = event
                    b_ix += 1

        except StopIteration:
            self.output_queue.put(None)

    def run_pipeline(self, b_ix, batch_indices, batch_slot, cuda_event):
        # print(b_ix, batch_indices)
        self.memory_context.start_batch(b_ix)
        args = []
        if IS_CUDA:
            stream = self.cuda_streams[batch_slot]
            ctx = ch.cuda.stream(stream)
        else:
            ctx = nullcontext()
        first_stage = False

        with ctx:
            if IS_CUDA:
                if cuda_event:
                    cuda_event.wait()
            for stage, banks in self.memory_bank_per_stage.items():
                args.insert(0, batch_indices)
                for bank in banks:
                    if bank is not None:
                        if isinstance(bank, tuple):
                            bank = tuple(x[batch_slot] for x in bank)
                        else:
                            bank = bank[batch_slot]
                    args.append(bank)
                args.append(self.metadata)
                args.append(self.storage_state)
                code = self.loader.code_per_stage[stage]
                result = code(*args)
                args = list(result)
                if first_stage:
                    first_stage = False
                    self.memory_context.end_batch(b_ix)
        return tuple(x[:len(batch_indices)] for x in args)

    def __next__(self):
        result = self.output_queue.get()
        if result is None:
            self.close()
            raise StopIteration()
        slot, result = result
        if IS_CUDA:
            stream = self.cuda_streams[slot]
            # We wait for the copy to be done
            ch.cuda.current_stream().wait_stream(stream)
        return result

    def __iter__(self):
        return self

    def close(self):
        self.terminate_event.set()
        if not self.closed:
            self.memory_context.__exit__(None, None, None)

    def __del__(self):
        self.close()
        
