import numpy as np

from .base import MemoryManager
from ..pipeline.compiler import Compiler

class RAMMemoryManager(MemoryManager):

    def schedule_epoch(self, schedule):
        return super().schedule_epoch(schedule)

    def __enter__(self):
        self.mmap = np.memmap(self.reader.file_name, 'uint8', mode='r')
        return super().__enter__()

    def __exit__(self, __exc_type, __exc_value, __traceback):
        # Numpy doesn't have an API to close memory maps yet
        # The only thing one can do is flush it be since we are not
        # Writing to it it's pointless
        return super().__exit__(__exc_type, __exc_value, __traceback)

    def compile_reader(self):
        ptrs = self.ptrs
        sizes = self.sizes
        mmap = self.mmap
        def read(address):
            six = np.searchsorted(ptrs, address)
            return mmap[address:address + sizes[six]]

        return Compiler.compile(read)

