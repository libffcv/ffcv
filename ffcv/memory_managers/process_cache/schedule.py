from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping
from queue import Queue

import numpy as np

from .page_reader import PageReader


@dataclass
class Schedule:
    # Number of slots needed
    num_slots: int
    # Which slot to use for each page
    page_to_slot: Mapping[int, int]
    # First iteration a page can be loaded
    can_prefetch_at: Mapping[int, int]
    # Iteration at which a page *has* to be loaded
    entering_at: Mapping[int, int]
    # Iteration at which we can discard a page
    leaving_at: Mapping[int, int]

def compute_schedule(pages_in_batch, prefetch_ahead = 3):
    # We determine what is the early and latest times we will need a page
    page_end = {}
    page_start = {}
    for b_id, pages in enumerate(pages_in_batch):
        for page in pages:
            page_end[page] = b_id
            if page not in page_start:
                page_start[page] = b_id

    # We determine which pages are
    # - Can be preloaded
    # - Are needed
    # - Can be diposed of
    # At a given batch
    entering_at = defaultdict(set)
    can_prefetch_at = defaultdict(set)
    leaving_at = defaultdict(set)
    for page in page_start.keys():
        prefetch_start = max(0, page_start[page] - prefetch_ahead)
        can_prefetch_at[prefetch_start].add(page)
        entering_at[page_start[page]].add(page)
        leaving_at[page_end[page] + 1].add(page)


     # We now find how many pages we need to keep in our buffer
     # We also determine where which page is going to reside
    next_slot = 0
    page_to_slot = {}
    free_slots = set()

    # For each batch
    for b_id in range(len(pages_in_batch)):
        # First we free the pages that are leaving
        for page in leaving_at[b_id]:
            free_slots.add(page_to_slot[page])

        # We use the prefetch timing here because we want to be able
        # To start prefetching ahead of time and not overwrite a slot
        # That is currently used
        for page in can_prefetch_at[b_id]:
            # Then we find a slot for the incoming pages
            if free_slots:
                # There is a slot available for this page
                slot = free_slots.pop()
            else:
                # We have to allocate a new slot because we ran out
                slot = next_slot
                next_slot += 1

            page_to_slot[page] = slot

    return Schedule(next_slot, page_to_slot,
                    can_prefetch_at, entering_at, leaving_at)

class ScheduleExecutor():

    def __init__(self, fname: str, schedule: Schedule,
                 memory: np.ndarray, num_workers: int=12):
        self.fname = fname
        self.schedule = schedule
        self.memory = memory
        self.queries = Queue()
        self.loaded_queue = Queue()
        self.num_workers = num_workers
        self.entered = False
        self.next_batch = 0
        self.loaded = set()

    def __enter__(self):
        msg = "You can only enter a ScheduleExecutor once"
        if self.entered:
            raise Exception(msg)
        self.entered = True
# Create the number of threads we were asked to
        threads = []
        for _ in range(self.num_workers):
            thread = PageReader(self.fname, self.queries,
                                self.loaded_queue, self.memory)
            thread.start()
            threads.append(thread)

        self.threads = threads

    def __exit__(self, *_):
        # Terminating the child threads
        for _ in range(self.num_workers):
            self.queries.put(None)

    def load_batch(self, current_batch):
        assert current_batch == self.next_batch

        # Start prefetching everything we are allowed to
        to_prefetch = self.schedule.can_prefetch_at[current_batch]
        for page_to_fetch in to_prefetch:
            q = (page_to_fetch, self.schedule.page_to_slot[page_to_fetch])
            self.queries.put(q)

        # Wait until we have all the pages we need
        to_wait_for = self.schedule.entering_at[current_batch]
        for page in to_wait_for:
            while page not in self.loaded:
                next_loaded = self.loaded_queue.get()
                self.loaded.add(next_loaded)

        # We enforce that we read in order otherwise our
        # assumptions are broken
        self.next_batch = current_batch + 1
