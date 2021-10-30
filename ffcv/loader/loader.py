import enum
from ffcv.pipeline.compiler import Compiler
from multiprocessing import cpu_count
from typing import Mapping, Optional, Sequence, TYPE_CHECKING, Union, Literal
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
from ..transforms.ops import ToTensor
from ..transforms.module import ModuleWrapper

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
                 batch_size: int,
                 num_workers: int = -1,
                 memory_manager: MEMORY_MANAGER_TYPE = MemoryManagerOption.RAM,
                 order: ORDER_TYPE = OrderOption.SEQUENTIAL,
                 distributed: bool = False,
                 seed: int = None,  # For ordering of samples
                 indices: Sequence[int] = None,  # For subset selection
                 pipelines: Mapping[str, Sequence[Union[Operation, ch.nn.Module]]] = {},
                 device: ch.device = ch.device('cpu')):

        self.fname: str = fname
        self.batch_size: int = batch_size
        self.seed: Optional[int] = seed
        self.reader: Reader = Reader(self.fname)
        self.num_workers: int = num_workers
        Compiler.set_num_threads(self.num_workers)

        if self.num_workers < 1:
            self.num_workers = cpu_count()

        if indices is None:
            self.indices = np.arange(self.reader.num_samples, dtype='uint64')
        else:
            self.indices = np.array(indices)

        if distributed:
            raise NotImplemented("Not implemented yet")

        self.memory_manager: MemoryManager = MEMORY_MANAGER_MAP[memory_manager](
            self.reader)
        self.traversal_order: TraversalOrder = ORDER_MAP[order](self)

        # TODO EXIT eventually
        self.memory_manager.__enter__()

        memory_read = self.memory_manager.compile_reader()
        self.next_epoch: int = 0

        self.pipelines = {}

        for f_ix, (field_name, field) in enumerate(self.reader.handlers.items()):
            DecoderClass = field.get_decoder_class()
            try:
                operations = pipelines[field_name]
                if not isinstance(operations[0], DecoderClass):
                    msg = "The first operation of the pipeline for "
                    msg += f"'{field_name}' has to be a subclass of "
                    msg += f"{DecoderClass}"
                    raise ValueError(msg)

            except KeyError:
                try:
                    operations = [
                        DecoderClass(),
                        ToTensor()
                    ]
                except Exception:
                    msg = f"Impossible to create a default pipeline"
                    msg += f"{field_name}, please define one manually"
                    raise ValueError(msg)

            for i, op in enumerate(operations):
                assert isinstance(op, (ch.nn.Module, Operation)), op
                if isinstance(op, ch.nn.Module):
                    operations[i] = ModuleWrapper(op)

            for op in operations:
                op.accept_globals(self.reader.metadata[f'f{f_ix}'],
                                  memory_read)

            self.pipelines[field_name] = Pipeline(operations)

    def close(self):
        self.memory_manager.__exit__(None, None, None)

    def __iter__(self):
        cur_epoch = self.next_epoch
        self.next_epoch += 1
        Compiler.set_num_threads(self.num_workers)
        order = self.traversal_order.sample_order(cur_epoch)
        return EpochIterator(self, cur_epoch, order)

    def __len__(self):
        # TODO handle drop_last
        return int(np.ceil(len(self.indices) / self.batch_size))
