from typing import Callable, TYPE_CHECKING, Tuple, Type
from dataclasses import replace

import numpy as np

from .base import Field, ARG_TYPE
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.compiler import Compiler
from ..pipeline.allocation_query import AllocationQuery
from ..libffcv import memcpy


class BytesDecoder(Operation):

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        max_size = self.metadata['size'].max()

        my_shape = (max_size,)
        return (
            replace(previous_state, jit_mode=True, shape=my_shape,
                    dtype='<u1'),
            AllocationQuery(my_shape, dtype='<u1')
        )

    def generate_code(self) -> Callable:
        mem_read = self.memory_read
        my_memcpy = Compiler.compile(memcpy)
        my_range = Compiler.get_iterator()
        def decoder(batch_indices, destination, metadata, storage_state):
            for dest_ix in my_range(batch_indices.shape[0]):
                source_ix = batch_indices[dest_ix]
                data = mem_read(metadata[source_ix]['ptr'], storage_state)
                my_memcpy(data, destination[dest_ix])
            return destination

        return decoder

class BytesField(Field):
    """
    A subclass of :class:`~ffcv.fields.Field` supporting variable-length byte
    arrays.

    Intended for use with data such as text or raw data which may not have a
    fixed size. Data is written sequentially while saving pointers and read by
    pointer lookup.

    The writer expects to be passed a 1D uint8 numpy array of variable length for each sample.
    """
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

    def get_decoder_class(self) -> Type[Operation]:
        return BytesDecoder
