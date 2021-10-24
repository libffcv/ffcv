from typing import Callable, TYPE_CHECKING, Tuple
from dataclasses import replace

import numpy as np

from .base import Field, ARG_TYPE
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.stage import Stage
from ..pipeline.allocation_query import AllocationQuery

if TYPE_CHECKING:
    from ..memory_managers.base import MemoryManager

class BasicDecoder(Operation):

    def __init__(self, dtype, memory:'MemoryManager'):
        self.dtype = dtype
        self.memory: 'MemoryManager' = memory
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        my_shape = (1,)
        return (
            replace(previous_state, jit_mode=True,
                    stage=Stage.INDIVIDUAL, shape=my_shape,
                    dtype=self.dtype),
            AllocationQuery(my_shape, dtype=self.dtype)
        )
    
    def generate_code(self) -> Callable:
        memory: self.memory
        def decoder(field, destination):
            destination[:] = field
            return destination

        return decoder

class FloatField(Field):
    def __init__(self):
        pass

    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype('<f8')

    @staticmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        return FloatField()

    def to_binary(self) -> ARG_TYPE:
        return np.zeros(1, dtype=ARG_TYPE)[0]

    def encode(self, destination, field, malloc):
        destination[0] = field
        
    def get_decoder(self, metadata: np.array, memory: 'MemoryManager') -> Operation:
        return BasicDecoder(np.dtype('f8'), memory)

class IntField(Field):
    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype('<i8')

    @staticmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        return IntField()

    def to_binary(self) -> ARG_TYPE:
        return np.zeros(1, dtype=ARG_TYPE)[0]

    def encode(self, destination, field, malloc):
        # We just allocate 1024bytes for fun
        destination[0] = field

    def get_decoder(self, metadata: np.array, memory: 'MemoryManager') -> Operation:
        return BasicDecoder(np.dtype('i8'), memory)

