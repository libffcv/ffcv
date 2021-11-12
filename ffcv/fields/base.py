from __future__ import annotations
from typing import Type
import numpy as np
from abc import ABC, abstractmethod

from ..pipeline.operation import Operation

ARG_TYPE = np.dtype([('', '<u1', 1024)])

class Field(ABC):
    """Abstract Base Class for implementing fields (e.g., images, integers).

    Each dataset entry is comprised of one or more fields (for example, standard
    object detection datasets may have one field for images, and one for
    bounding boxes). Fields are responsible for implementing encoder and decoder
    function that determine how they will be written and read from the dataset file,
    respectively.

    See `here`<TODO>_ for information on how to implement a subclass of Field.
    """
    @property
    @abstractmethod
    def metadata_type(self) -> np.dtype:
        raise NotImplemented

    @staticmethod
    @abstractmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        raise NotImplementedError

    @abstractmethod
    def to_binary(self) -> ARG_TYPE:
        raise NotImplementedError

    @abstractmethod
    def encode(field, metadata_destination, malloc):
        raise NotImplementedError
    
    @abstractmethod
    def get_decoder_class(self) -> Type[Operation]:
        raise NotImplementedError
        