import numpy as np
import numba as nb

from .base import MemoryManager
from ..pipeline.compiler import Compiler

class RAMMemoryManager(MemoryManager):

    def schedule_epoch(self, schedule):
        return self

    def __enter__(self):
        self.mmap = np.memmap(self.reader.file_name, 'uint8', mode='r')
        return super().__enter__()

    @property
    def state(self):
        return (self.mmap, self.ptrs, self.sizes)

    @property
    def state_type(self):
        t1 = nb.uint8[::1]
        t1.multable = False
        t2 = nb.uint64[::1]
        t1.mutable = False
        return nb.types.Tuple([t1, t2, t2])

    def __exit__(self, __exc_type, __exc_value, __traceback):
        # Numpy doesn't have an API to close memory maps yet
        # The only thing one can do is flush it be since we are not
        # Writing to it it's pointless
        return super().__exit__(__exc_type, __exc_value, __traceback)

    def compile_reader(self):
        def read(address, mem_state):
            size = mem_state[2][np.searchsorted(mem_state[1], address)]
            return mem_state[0][address:address + size]

        return Compiler.compile(read, nb.uint8[::1](nb.uint64, self.state_type))

