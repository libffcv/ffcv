import json

import torch as ch
import numpy as np

from .bytes import BytesField

ENCODING = 'utf8'
SEPARATOR = '\0'  # Null byte

class JSONField(BytesField):

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
        if isinstance(batch, ch.Tensor):
            batch = batch.numpy()
        result = []
        for b in batch:
            sep_location = np.where(b == ord(SEPARATOR))[0][0]
            b = b[:sep_location]
            string = b.tobytes().decode(ENCODING)
            result.append(json.loads(string))
        return result
