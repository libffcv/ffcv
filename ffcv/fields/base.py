from __future__ import annotations
import numpy as np
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