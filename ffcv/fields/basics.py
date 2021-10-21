from typing import Callable, Tuple
from dataclasses import replace

import numpy as np

from .base import Field, ARG_TYPE
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.stage import Stage
from ..pipeline.allocation_query import AllocationQuery

class BasicDecoder(Operation):

    def __init__(self, dtype):
        self.dtype = dtype
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        return (
            replace(previous_state, jit_mode=True, stage=Stage.INDIVIDUAL),
            AllocationQuery((1,), dtype=self.dtype)
        )
    
    def generate_code(self) -> Callable:
        def decoder(field, destination, memory):
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
        
    def get_decoder(self, metadata: np.array) -> Operation:
        return BasicDecoder(np.float64)

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

    def get_decoder(self, metadata: np.array) -> Operation:
        return BasicDecoder(np.int64)

