from typing import Callable, TYPE_CHECKING, Tuple, Type
import warnings
import json
from dataclasses import replace

import numpy as np
import torch as ch

from .base import Field, ARG_TYPE
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.compiler import Compiler
from ..pipeline.allocation_query import AllocationQuery
from ..libffcv import memcpy

if TYPE_CHECKING:
    from ..memory_managers.base import MemoryManager

class NDArrayDecoder(Operation):
    """
    Default decoder for :class:`~ffcv.fields.NDArrayField`.
    """

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, AllocationQuery]:
        return (
            replace(previous_state, jit_mode=True,
                    shape=self.field.shape,
                    dtype=self.field.dtype),
            AllocationQuery(self.field.shape, self.field.dtype)
        )

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        mem_read = self.memory_read
        my_memcpy = Compiler.compile(memcpy)

        def decoder(indices, destination, metadata, storage_state):
            for ix in my_range(indices.shape[0]):
                sample_id = indices[ix]
                ptr = metadata[sample_id]
                data = mem_read(ptr, storage_state)
                my_memcpy(data, destination[ix].view(np.uint8))
            return destination

        return decoder

NDArrayArgsType = np.dtype([
    ('shape', '<u8', 32),  # 32 is the max number of dimensions for numpy
    ('type_length', '<u8'),  # length of the dtype description
])

class NDArrayField(Field):
    """A subclass of :class:`~ffcv.fields.Field` supporting
    multi-dimensional fixed size matrices of any numpy type.
    """
    def __init__(self, dtype:np.dtype, shape:Tuple[int, ...]):
        self.dtype = dtype
        self.shape = shape
        self.element_size = dtype.itemsize * np.prod(shape)
        if dtype == np.uint16:
            warnings.warn("Pytorch currently doesn't support uint16"
            "we recommend storing as int16 and reinterpret your data later"
            "in your pipeline")

    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype('<u8')

    @staticmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        header_size = NDArrayArgsType.itemsize
        header = binary[:header_size].view(NDArrayArgsType)[0]
        type_length = header['type_length']
        type_data = binary[header_size:][:type_length].tobytes().decode('ascii')
        type_desc = json.loads(type_data)
        type_desc = [tuple(x) for x in type_desc]
        assert len(type_desc) == 1
        dtype = np.dtype(type_desc)['f0']
        shape = list(header['shape'])
        while shape[-1] == 0:
            shape.pop()

        return NDArrayField(dtype, tuple(shape))

    def to_binary(self) -> ARG_TYPE:
        result = np.zeros(1, dtype=ARG_TYPE)[0]
        header = np.zeros(1, dtype=NDArrayArgsType)
        s = np.array(self.shape).astype('<u8')
        header['shape'][0][:len(s)] = s
        encoded_type = json.dumps(self.dtype.descr)
        encoded_type = np.frombuffer(encoded_type.encode('ascii'), dtype='<u1')
        header['type_length'][0] = len(encoded_type)
        to_write = np.concatenate([header.view('<u1'), encoded_type])
        result[0][:to_write.shape[0]] = to_write
        return result

    def encode(self, destination, field, malloc):
        destination[0], data_region = malloc(self.element_size)
        data_region[:] = field.reshape(-1).view('<u1')

    def get_decoder_class(self) -> Type[Operation]:
        return NDArrayDecoder


class TorchTensorField(NDArrayField):
    """A subclass of :class:`~ffcv.fields.Field` supporting
    multi-dimensional fixed size matrices of any torch type.
    """
    def __init__(self, dtype:ch.dtype, shape:Tuple[int, ...]):
        self.dtype = dtype
        self.shape = shape
        dtype = ch.zeros(0, dtype=dtype).numpy().dtype

        super().__init__(dtype, shape)


    def encode(self, destination, field, malloc):
        field = field.numpy()
        return super().encode(destination, field, malloc)
