from threading import Thread
from queue import Queue

import numpy as np

from ...libffcv import read


class PageReader(Thread):

    def __init__(self, fname:str, queries: Queue, loaded: Queue,
                 memory: np.ndarray):
        self.fname: str = fname
        self.queries: Queue = queries
        self.memory: np.ndarray = memory
        self.page_size = memory.shape[1]
        self.loaded: Queue = loaded
        super().__init__()

    def run(self):
        with open(self.fname, 'rb') as handle:
            fileno = handle.fileno()

            while True:
                query = self.queries.get()
                # No more work
                if query is None:
                    break

                page_number, slot = query
                offset = np.uint64(page_number * self.page_size)
                read(fileno, self.memory[slot], offset)
                self.loaded.put(page_number)

