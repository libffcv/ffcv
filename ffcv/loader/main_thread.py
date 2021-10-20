from multiprocessing import cpu_count
from typing import Mapping, Sequence, Union, Literal
from enum import Enum, unique, auto

import torch as ch
import numpy as np

from ..memory_managers.ram import RAMMemoryManager
from ..memory_managers.base import MemoryManager
from ..reader import Reader
from ..traversal_order.base import TraversalOrder
from ..traversal_order import Random, Sequential


@unique
class MemoryManagerOption(Enum):
    RAM = auto()


@unique
class OrderOption(Enum):
    SEQUENTIAL = auto()
    RANDOM = auto()


MEMORY_MANAGER_TYPE = Literal[MemoryManagerOption.RAM]

ORDER_TYPE = Union[
    TraversalOrder,
    Literal[OrderOption.SEQUENTIAL,
            OrderOption.RANDOM]

]
MEMORY_MANAGER_MAP: Mapping[MEMORY_MANAGER_TYPE, MemoryManager] = {
    MemoryManagerOption.RAM: RAMMemoryManager
}

ORDER_MAP: Mapping[ORDER_TYPE, TraversalOrder] = {
    OrderOption.RANDOM: Random,
    OrderOption.SEQUENTIAL: Sequential
}


class Loader:

    def __init__(self,
                 fname: str,
                 num_workers: int = -1,
                 memory_manager: MEMORY_MANAGER_TYPE = MemoryManagerOption.RAM,
                 order: ORDER_TYPE = OrderOption.SEQUENTIAL,
                 distributed: bool = False,
                 seed: int = 42,  # For ordering of samples
                 indices: Sequence[int] = None,  # For subset selection
                 device: ch.device = ch.device('cpu')):

        self.fname = fname
        self.seed = seed
        self.reader = Reader(self.fname)

        if num_workers < 1:
            self.num_workers = cpu_count()
        else:
            self.num_workers = num_workers
            
        if indices is None:
            self.indices = np.arange(self.reader.num_samples, dtype='uint64')
        else:
            self.indices = np.array(indices)
            
        if distributed:
            raise NotImplemented("Not implemented yet")
        
            
        self.memory_manager: MemoryManager = MEMORY_MANAGER_MAP[memory_manager](self.reader)
        self.traversal_order: TraversalOrder = ORDER_MAP[order](self)
        
        self.next_epoch = 0
        
    def __iter__(self):
        cur_epoch = self.next_epoch
        print(self.traversal_order.sample_order(cur_epoch))
        self.next_epoch += 1
        return self
    
    def __next__(self):
        raise StopIteration()
        pass
    
    