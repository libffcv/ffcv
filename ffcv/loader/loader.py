from multiprocessing import cpu_count
from typing import Mapping, Sequence, Union, Literal
from enum import Enum, unique, auto

import torch as ch
import numpy as np

from .epoch_iterator import EpochIterator
from ..memory_managers.ram import RAMMemoryManager
from ..memory_managers.base import MemoryManager
from ..reader import Reader
from ..traversal_order.base import TraversalOrder
from ..traversal_order import Random, Sequential
from ..pipeline import Pipeline
from ..pipeline.operation import Operation
from ..fields.base import Field


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

class Pipelines():
    def __init__(self, fields: Mapping[str, Field]):
        self.fields: Mapping[str, Field] = fields
        
        self.decoders: Mapping[str, Operation] = {
            k: self.fields[k].get_decoder() for k in self.fields
        }

        self.pipelines: Mapping[str, Pipeline] = {
            k: Pipeline([self.decoders[k]]) for k in self.fields.keys()
        }

    def __setitem__(self, name: str, value: Sequence[Operation]) -> None:
        if name not in self.pipelines:
            raise KeyError(f"Unknown field: {name}")
        
        self.pipelines[name] = Pipeline([self.decoders[name], *value])
        
    def __getitem__(self, key: str):
        return self.pipelines[key]

class Loader:

    def __init__(self,
                 fname: str,
                 batch_size: int,
                 num_workers: int = -1,
                 memory_manager: MEMORY_MANAGER_TYPE = MemoryManagerOption.RAM,
                 order: ORDER_TYPE = OrderOption.SEQUENTIAL,
                 distributed: bool = False,
                 seed: int = 42,  # For ordering of samples
                 indices: Sequence[int] = None,  # For subset selection
                 device: ch.device = ch.device('cpu')):

        self.fname: str = fname
        self.batch_size:int = batch_size
        self.seed: int = seed
        self.reader: Reader = Reader(self.fname)
        self.num_workers: int = num_workers

        if self.num_workers < 1:
            self.num_workers = cpu_count()
            
        if indices is None:
            self.indices = np.arange(self.reader.num_samples, dtype='uint64')
        else:
            self.indices = np.array(indices)
            
        if distributed:
            raise NotImplemented("Not implemented yet")
        
            
        self.memory_manager: MemoryManager = MEMORY_MANAGER_MAP[memory_manager](self.reader)
        self.traversal_order: TraversalOrder = ORDER_MAP[order](self)
        
        # TODO EXIT eventually
        self.memory_manager.__enter__()
        
        self.next_epoch: int = 0
        self.pipelines: Pipelines = Pipelines(self.reader.handlers)
        
    def __iter__(self):
        cur_epoch = self.next_epoch
        self.next_epoch += 1
        order = self.traversal_order.sample_order(cur_epoch)
        return EpochIterator(self, cur_epoch, order)
    