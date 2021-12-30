from collections import defaultdict

import numpy as np

from ..base import MemoryManager, MemoryContext
from ..common import BATCHES_TYPE
from .schedule import Schedule, ScheduleExecutor, compute_schedule


class ProcessCacheContext(MemoryContext):

    def __init__(self, manager: MemoryManager, batches: BATCHES_TYPE):
        self.manager = manager
        self.fname = manager.reader.file_name
        self.batches = batches
        self.page_size = manager.reader.page_size

    @property
    def state(self):
        return (self.memory, self.manager.ptrs,
                self.manager.sizes, self.page_to_slot)

    def __enter__(self):
        pages_at_batch = []
        for batch in self.batches:
            pages_needed = set()
            for sample_id in batch:
                pages_needed.update(self.manager.sample_to_pages[sample_id])
            pages_at_batch.append(pages_needed)

        self.schedule = compute_schedule(pages_at_batch)
        self.memory = np.zeros((self.schedule.num_slots, self.page_size),
                               dtype='<u1')
        self.executor = ScheduleExecutor(self.fname,
                                         self.schedule,
                                         self.memory)

        try:
            max_page = max(self.schedule.page_to_slot.keys())
        except ValueError:
            max_page = -1

        # We need max + 1 slots
        # We use a table as it's O(1) indexing. Pages for the header will
        # be unused however so we are losing some space
        self.page_to_slot = np.zeros(max_page + np.uint32(1), dtype=np.uint32)

        for p, s in self.schedule.page_to_slot.items():
            self.page_to_slot[p] = s

        self.executor.__enter__()

    def start_batch(self, batch: int):
        self.executor.load_batch(batch)
        return super().start_batch(batch)


    def __exit__(self, *args):
        self.executor.__exit__(*args)
