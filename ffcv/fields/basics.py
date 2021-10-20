from typing import Callable
from dataclasses import replace

import numpy as np

from .base import Field, ARG_TYPE
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.stage import Stage

class BasicDecoder(Operation):
    
    def advance_state(self, previous_state: State) -> State:
        return replace(previous_state, jit_mode=True, stage=Stage.INDIVIDUAL)
    
    def generate_code(self) -> Callable:
        def decoder(field, memory):
            return field[0]
        
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
        
    def get_decoder(self) -> Operation:
        return BasicDecoder()

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

    def get_decoder(self) -> Operation:
        return BasicDecoder()

