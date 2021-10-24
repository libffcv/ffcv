from typing import Callable, TYPE_CHECKING, Tuple
from dataclasses import replace

import numpy as np

from .base import Field, ARG_TYPE
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.stage import Stage
from ..pipeline.allocation_query import AllocationQuery


class BytesDecoder(Operation):

    def __init__(self, dtype, metadata: np.ndarray, memory_read):
        self.dtype = dtype
        self.memory_read = memory_read
        self.metadata = metadata

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        max_size = self.metadata['size'].max()
        my_shape = (max_size,)
        return (
            replace(previous_state, jit_mode=True,
                    stage=Stage.INDIVIDUAL, shape=my_shape,
                    dtype='<u1'),
            None
        )

    def generate_code(self) -> Callable:
        read = self.memory_read
        def decoder(field, _):
            return read(field['ptr'])

        return decoder

class BytesField(Field):
    def __init__(self):
        pass

    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('ptr', '<u8'),
            ('size', '<u8')
        ])

    @staticmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        return BytesField()

    def to_binary(self) -> ARG_TYPE:
        return np.zeros(1, dtype=ARG_TYPE)[0]

    def encode(self, destination, field, malloc):
        ptr, buffer = malloc(field.size)
        buffer[:] = field
        destination['ptr'] = ptr
        destination['size'] = field.size

    def get_decoder(self, metadata: np.array, read_function) -> Operation:
        return BytesDecoder(np.dtype('f8'), metadata, read_function)