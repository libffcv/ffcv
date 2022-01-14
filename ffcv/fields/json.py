import json

import torch as ch
import numpy as np

from .bytes import BytesField

ENCODING = 'utf8'
SEPARATOR = '\0'  # Null byte

class JSONField(BytesField):
    """A subclass of :class:`~ffcv.fields.BytesField` that encodes JSON data.

    The writer expects to be passed a dict that is compatible with the JSON specification.

    .. warning ::
        Because FFCV is based on tensors/ndarrays the reader and therefore the loader can't give return JSON to the user. This is why we provide :class:`~ffcv.fields.JSONField.unpack` which does the conversion. It's up to the user to call it in the main body of the loop

    """

    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('ptr', '<u8'),
            ('size', '<u8')
        ])

    def encode(self, destination, field, malloc):
        # Add null terminating byte
        content = (json.dumps(field) + SEPARATOR).encode(ENCODING)
        field = np.frombuffer(content, dtype='uint8')
        return super().encode(destination, field, malloc)

    @staticmethod
    def unpack(batch):
        """Convert back the output of a :class:`~ffcv.fields.JSONField` field produced by :class:`~ffcv.Loader` into an actual JSON.
        
        It works both on an entire batch and will return an array of python dicts or a single sample and will simply return a dict.
        """
        if isinstance(batch, ch.Tensor):
            batch = batch.numpy()

        single_instance = len(batch.shape) == 1
        if single_instance:
            batch = [batch]

        result = []
        for b in batch:
            sep_location = np.where(b == ord(SEPARATOR))[0][0]
            b = b[:sep_location]
            string = b.tobytes().decode(ENCODING)
            result.append(json.loads(string))

        if single_instance:
            result = result[0]

        return result
