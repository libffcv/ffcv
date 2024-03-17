from typing import TYPE_CHECKING

import numpy as np
import numba as nb

from .base import MemoryManager, MemoryContext
from ..pipeline.compiler import Compiler

if TYPE_CHECKING:
    from ..reader import Reader

from ffcv.libffcv import init_client, get_slice
import threading
init_client=nb.njit(init_client)
class NetContext(MemoryContext):
    def __init__(self, manager:MemoryManager, ):
        
        self.manager = manager
        self.mmap = np.memmap(self.manager.reader.file_name,
                                  'uint8', mode='r')
        

    @property
    def state(self):
        return (self.mmap, self.manager.ptrs, self.manager.sizes)
    
    def thread_state(self):
        print("insert thread state ", threading.current_thread())
        sockfd: int = init_client()
        buffer = np.zeros(int(self.manager.sizes.max())+1, dtype=np.uint8)
        return (sockfd,buffer)

    def __enter__(self):
        res = super().__enter__()
        return res

    def __exit__(self, __exc_type, __exc_value, __traceback):
        # Numpy doesn't have an API to close memory maps yet
        # The only thing one can do is flush it be since we are not
        # Writing to it it's pointless
        # Moreover we want to avoid opening the memmap over and over
        # anyway.
        return super().__exit__(__exc_type, __exc_value, __traceback)


class NetCacheManager(MemoryManager):

    def __init__(self, reader: 'Reader'):
        super().__init__(reader)
        self.context = NetContext(self)

    def schedule_epoch(self, schedule):
        return self.context
    
    @property
    def state_type(self):
        t1 = nb.uint8[::1]
        t1.multable = False
        t2 = nb.uint64[::1]
        t1.mutable = False
        return nb.types.Tuple([t1, t2, t2,nb.int32, t1])

    def compile_reader(self):
        c_get_slice = nb.njit(get_slice)
        # buffer = self.context.buffer
        
        def read(address, mem_state):
            mmap, ptrs, sizes, sockfd, buffer = mem_state
            size = sizes[np.searchsorted(ptrs, address)]
            if len(mem_state[0])<size:
                print("Error: size of mmap is smaller than the size of the slice", size, len(mem_state[0]))

            c_get_slice(sockfd, address, address + size, buffer)
            get_data = buffer[:size]
            ref_data = mmap[address:address + size]
            
            error = (get_data != ref_data).sum()
            assert error==0, f"Data from NetCache is not equal to the data from the file: {address}, {size}, {error}"
            
            return get_data
        
        return Compiler.compile(read, nb.uint8[::1](nb.uint64, self.state_type))
        

