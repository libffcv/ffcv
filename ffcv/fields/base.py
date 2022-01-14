from __future__ import annotations
from typing import Type
import numpy as np
from abc import ABC, abstractmethod

from ..pipeline.operation import Operation

ARG_TYPE = np.dtype([('', '<u1', 1024)])

class Field(ABC):
    """
    Abstract Base Class for implementing fields (e.g., images, integers).

    Each dataset entry is comprised of one or more fields (for example, standard
    object detection datasets may have one field for images, and one for
    bounding boxes). Fields are responsible for implementing encode and get_decoder_class
    functions that determine how they will be written and read from the dataset file,
    respectively.

    .. note ::
        It is possible to have multiple potential decoder for a given Field. ``RGBImageField`` one example. Users can specify which decoder to use in the ``piplines`` argument of the ``Loader`` class.

    See :ref:`here <TODO>` for information on how to implement a subclass of Field.
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
