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

def select_buffer(buffer, batch_slot, count):
    """Util function to select the relevent subpart of a buffer for a given
    batch_slot and batch size"""
    if buffer is None:
        return None
    if isinstance(buffer, tuple):
        return tuple(select_buffer(x, batch_slot, count) for x in buffer)

    return buffer[batch_slot][:count]


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

        if IS_CUDA:
            self.current_stream = ch.cuda.current_stream()

        try:
            self.memory_context.__enter__()
        except MemoryError as e:
            if loader.traversal_order != QuasiRandom:
                print(QUASIRANDOM_ERROR_MSG)
                print('Full error below:')

            raise e

        self.storage_state = self.memory_context.state

        self.cuda_streams = [(ch.cuda.Stream() if IS_CUDA else None)
                             for _ in range(self.loader.batches_ahead + 2)]

        self.memory_allocations = self.loader.graph.allocate_memory(
            self.loader.batch_size,
            self.loader.batches_ahead + 2
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
                # print("RES", b_ix, "ready")
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
                    # print("SUB", b_ix)
                    # We were able to submit this batch
                    # Therefore it means that the user must have entered the for loop for
                    # (batch_slot - batch_ahead + 1) % (batches ahead + 2)
                    # Therefore batch_slot - batch_ahead must have all it's work submitted
                    # We will record an event of all the work submitted on the main stream
                    # and make sure no one overwrite the data until they are done
                    just_finished_slot = (slot - self.loader.batches_ahead - 1) % (self.loader.batches_ahead + 2)
                    # print("JFS", just_finished_slot)
                    event = ch.cuda.Event()
                    event.record(self.current_stream)
                    events[just_finished_slot] = event
                b_ix += 1

        except StopIteration:
            self.output_queue.put(None)

    def run_pipeline(self, b_ix, batch_indices, batch_slot, cuda_event):
        self.memory_context.start_batch(b_ix)
        args = []
        if IS_CUDA:
            stream = self.cuda_streams[batch_slot]
            ctx = ch.cuda.stream(stream)
        else:
            ctx = nullcontext()
        first_stage = False


        code, outputs = self.loader.code
        with ctx:
            if IS_CUDA:
                if cuda_event:
                    cuda_event.wait()

            args = {
                'batch_indices': batch_indices,
                'storage_state': self.storage_state,
                'metadata': self.metadata,
                **{
                    f'memory_{k}':select_buffer(v, batch_slot, len(batch_indices))
                    for (k, v) in self.memory_allocations['operation'].items()
                },
                **{
                    f'shared_memory_{k}': select_buffer(v, batch_slot, len(batch_indices))
                    for (k, v) in self.memory_allocations['shared'].items()
                }
            }

            for stage_code, define_outputs in code:
                results = stage_code(**args)
                for node_id, result in zip(define_outputs, results):
                    args[f'result_{node_id}'] = result
                pass

            result = tuple(args[f'result_{x}'] for x in outputs)
            return result

    def __next__(self):
        result = self.output_queue.get()
        if result is None:
            self.close()
            raise StopIteration()
        slot, result = result
        if IS_CUDA:
            stream = self.cuda_streams[slot]
            # We wait for the copy to be done
            self.current_stream.wait_stream(stream)
        return result

    def __iter__(self):
        return self

    def close(self):
        self.terminate_event.set()
        if not self.closed:
            self.memory_context.__exit__(None, None, None)

    def __del__(self):
        self.close()
        
