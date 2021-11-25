from collections import defaultdict
from threading import Thread, Event
from queue import Queue, Full
from typing import Sequence, TYPE_CHECKING

import torch as ch

from ..utils import chunks

if TYPE_CHECKING:
    from .loader import Loader


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
        self.memory_context.__enter__()
        self.storage_state = self.memory_context.state

        self.memory_bank_per_stage = defaultdict(list)

        self.cuda_streams = [ch.cuda.Stream()
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
        try:
            b_ix = 0
            while True:
                ixes = next(self.iter_ixes)
                slot = self.current_batch_slot
                self.current_batch_slot = (
                    slot + 1) % (self.loader.batches_ahead + 2)
                result = self.run_pipeline(b_ix, ixes, slot)
                to_output = (slot, result)
                while True:
                    try:
                        self.output_queue.put(to_output, block=True, timeout=0.5)
                        break
                    except Full:
                        pass

                    if self.terminate_event.is_set():
                        return
                b_ix += 1

        except StopIteration:
            self.output_queue.put(None)

    def run_pipeline(self, b_ix, batch_indices, batch_slot):
        # print(b_ix, batch_indices)
        self.memory_context.start_batch(b_ix)
        args = [batch_indices]
        stream = self.cuda_streams[batch_slot]
        stream.synchronize()
        first_stage = False
        with ch.cuda.stream(stream):
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
                if first_stage:
                    first_stage = False
                    self.memory_context.end_batch(b_ix)
        return tuple(args)

    def __next__(self):
        result = self.output_queue.get()
        if result is None:
            self.close()
            raise StopIteration()
        slot, result = result
        # We wait for the copy to be done
        return result

    def __iter__(self):
        return self

    def close(self):
        self.terminate_event.set()
        if not self.closed:
            self.memory_context.__exit__(None, None, None)

    def __del__(self):
        self.close()
        