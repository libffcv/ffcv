import numpy as np
from numba import carray
import numba as nb
import ctypes
from ..utils import cast_int_to_byte_ptr
from ctypes import c_uint64


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
        ptr_p = self.sizes.ctypes.data
        sizes_p = self.sizes.ctypes.data
        mmap_p = self.mmap.ctypes.data

        num_entries = len(self.ptrs)
        mmap_size = self.mmap.shape[0]
        
        u64 = ctypes.c_uint64

        def read(address):
            # Conversion of pointers
            sizes = carray(cast_int_to_byte_ptr(sizes_p), shape=(num_entries, )).view(np.uint64)
            ptrs = carray(cast_int_to_byte_ptr(ptr_p), shape=(num_entries, )).view(np.uint64)
            size = sizes[np.searchsorted(ptrs, address)]

            mmap = carray(cast_int_to_byte_ptr(mmap_p), shape=(num_entries, ))
            return mmap[address:address + size]

        return Compiler.compile(read)

