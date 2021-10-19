import numpy as np

from .base import MemoryManager

class RAMMemoryManager(MemoryManager):

    def schedule_epoch(self, schedule):
        return super().schedule_epoch(schedule)

    def __enter__(self):
        self.mmap = np.memmap(self.dataset_path, 'uint8', mode='r')
        return super().__enter__()

    def __exit__(self, __exc_type, __exc_value, __traceback):
        # Numpy doesn't have an API to close memory maps yet
        # The only thing one can do is flush it be since we are not
        # Writing to it it's pointless
        return super().__exit__(__exc_type, __exc_value, __traceback)

    def read(self, address, length):
        return self.mmap[address:address + length]

