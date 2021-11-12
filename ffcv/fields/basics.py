from typing import Callable, TYPE_CHECKING, Tuple, Type
from dataclasses import replace

import numpy as np

from .base import Field, ARG_TYPE
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.allocation_query import AllocationQuery

if TYPE_CHECKING:
    from ..memory_managers.base import MemoryManager

class BasicDecoder(Operation):

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        my_shape = (1,)
        return (
            replace(previous_state, jit_mode=True,
                    shape=my_shape,
                    dtype=self.dtype),
            AllocationQuery(my_shape, dtype=self.dtype)
        )
    
    def generate_code(self) -> Callable:
        def decoder(indices, destination, metadata, storage_state):
            for ix, sample_id in enumerate(indices):
                destination[ix] = metadata[sample_id]
            return destination

        return decoder
        
class IntDecoder(BasicDecoder):
    dtype = np.dtype('<i8')

class FloatDecoder(BasicDecoder):
    dtype = np.dtype('<f8')

class FloatField(Field):
    """
    A subclass of :class:`~ffcv.fields.Field` supporting (scalar) floating-point
    values.
    """
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
        
    def get_decoder_class(self) -> Type[Operation]:
        return FloatDecoder

class IntField(Field):
    """
    A subclass of :class:`~ffcv.fields.Field` supporting (scalar) floating-point
    values.
    """
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

    def get_decoder_class(self) -> Type[Operation]:
        return IntDecoder

