from typing import TYPE_CHECKING

import numpy as np
import numba as nb

from .base import MemoryManager, MemoryContext
from ..pipeline.compiler import Compiler

if TYPE_CHECKING:
    from ..reader import Reader

from multiprocessing.shared_memory import SharedMemory
import torch.distributed as dist

class SharedMemoryContext(MemoryContext):
    def __init__(self, manager:MemoryManager, ):
        
        self.manager = manager
        file_name = self.manager.reader.file_name
        name= file_name.split('/')[-1]
        print("loading", name)
        self.mmap = np.memmap(file_name, 'uint8', mode='r')
        size= len(self.mmap)
        
        if dist.is_initialized():
            if dist.get_rank()==0:
                mem = SharedMemory(name=name, create=True, size=size)
            else:
                mem = SharedMemory(name=name, create=False, size=size)            
        else:
            mem = SharedMemory(name=name, create=True, size=size)
            
        self.mmap = np.frombuffer(mem.buf, dtype=np.uint8)
        if dist.is_initialized():
            if dist.get_rank()==0:
                self.mmap[:] = np.fromfile(file_name, 'uint8')
            dist.barrier()
        else:
            self.mmap[:] = np.fromfile(file_name, 'uint8')

    @property
    def state(self):
        return (self.mmap, self.manager.ptrs, self.manager.sizes)
    

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


class SharedMemoryManager(MemoryManager):

    def __init__(self, reader: 'Reader'):
        super().__init__(reader)
        self.context = SharedMemoryContext(self)

    def schedule_epoch(self, schedule):
        return self.context
    
    @property
    def state_type(self):
        t1 = nb.uint8[::1]
        t1.multable = False
        t2 = nb.uint64[::1]
        t1.mutable = False
        return nb.types.Tuple([t1, t2, t2])

    def compile_reader(self):
        def read(address, mem_state):
            mmap, ptrs, sizes = mem_state
            size = sizes[np.searchsorted(ptrs, address)]
            ref_data = mmap[address:address + size]
            return ref_data
        
        return Compiler.compile(read, nb.uint8[::1](nb.uint64, self.state_type))
        

