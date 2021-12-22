from ast import Bytes
import json

import numpy as np

from .bytes import BytesField


class JSONField(BytesField):

    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('ptr', '<u8'),
            ('size', '<u8')
        ])

    def encode(self, destination, field, malloc):
       # Add null terminating byte
       content = (json.dumps(field) + '\0').encode('utf8')
       field = np.frombuffer(content, dtype='uint8')
       return super().encode(destination, field, malloc)
