from __future__ import annotations
import numpy as np
from collections import namedtuple
from abc import ABC, abstractmethod

ARG_TYPE = np.dtype('<u1', 1024)

class Field(ABC):

    @property
    @abstractmethod
    def metadata_type(self) -> np.dtype:
        raise NotImplemented

    @staticmethod
    @abstractmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        raise NotImplemented()

    @abstractmethod
    def to_binary(self) -> ARG_TYPE:
        raise NotImplemented()

    @abstractmethod
    def encode(field, metadata_destination, malloc):
        raise NotImplemented()


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

class IntField(Field):
    def __init__(self):
        pass

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
        ptr, buff = malloc(1024)
        buff[:] = field + 17
        destination[0] = field