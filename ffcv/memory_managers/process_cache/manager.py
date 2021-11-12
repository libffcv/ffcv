import numba as nb
import numpy as np

from .context import ProcessCacheContext
from ...pipeline.compiler import Compiler
from ..base import MemoryManager, MemoryContext
from ..common import BATCHES_TYPE

class ProcessCacheManager(MemoryManager):

    def schedule_epoch(self, batches: BATCHES_TYPE) -> MemoryContext:
        return ProcessCacheContext(self, batches)

    @property
    def state_type(self):
        # The data
        t1 = nb.uint8[:, ::1]
        t1.mutable = False

        # The pointers
        t2 = nb.uint64[::1]
        t2.mutable = False
        #
        # Their size
        t3 = nb.uint64[::1]
        t3.mutable = False

        # Page to slot
        t4 = nb.uint32[::1]
        t4.mutable = False

        return nb.types.Tuple([t1, t2, t3, t4])

    def compile_reader(self):
        page_size = self.reader.page_size
        page_size_log2 = np.uint32(np.log2(page_size))

        def read(address, mem_state):
            size = mem_state[2][np.searchsorted(mem_state[1], address)]
            page = address >> page_size_log2
            offset = address - (page << page_size_log2)
            page_slot = mem_state[3][page]
            return mem_state[0][page_slot, offset:offset + size]

        return Compiler.compile(read)
