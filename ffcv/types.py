from typing import List

import numpy as np


from .fields.base import Field
from .fields import (
    FloatField, IntField, RGBImageField,
    BytesField, NDArrayField, JSONField,
    TorchTensorField
)

CURRENT_VERSION = 2

# Note that in this file we use dtypes in the format <u4 indead of uint32. This
# forces endinaness of the data making datasets compatible between different
# CPU architectures ARM for example is big-endian and x86 is little endian. We
# fix the endianness to little-endian as we suspect that this library will
# mostly be used on x86 architecture (at least in the near future)

# Type describing the data coming
HeaderType = np.dtype([
    ('version', '<u2'),
    ('num_fields', '<u2'),
    ('page_size', '<u4'),
    ('num_samples', '<u8'),
    ('alloc_table_ptr', '<u8')
], align=True)

ALLOC_TABLE_TYPE = np.dtype([
    ('sample_id', '<u8'),
    ('ptr', '<u8'),
    ('size', '<u8'),
])

FieldDescType = np.dtype([
    # This identifier will inform us on how to decode that field
    ('type_id', '<u1'),
    ('name', ('<u1', 16)),
    # Data that will depend on the type of the field (some might need arguments
    # like images, but some might not like integers and floats)
    ('arguments', ('<u1', (1024, )))
], align=True)

# Map from type_id to the handler for that kind of data
TYPE_ID_HANDLER = {
    255 : None,
    0   : FloatField,
    1   : IntField,
    2   : RGBImageField,
    3   : BytesField,
    4   : NDArrayField,
    5   : JSONField,
    6   : TorchTensorField
}

# Parse the fields descriptors from the header of the dataset
# Return the corresponding handlers

def get_handlers(field_descriptors):
    handlers = []
    for field_descriptor in field_descriptors:
        type_id = field_descriptor['type_id']
        Handler = TYPE_ID_HANDLER[type_id]
        if Handler is None:
            handlers.append(None)
        else:
            handlers.append(Handler.from_binary(field_descriptor['arguments']))
    return handlers



# From a list of handlers return the combined data type that will
# describe a complete sample
def get_metadata_type(handlers: List[Field]) -> np.dtype:
    return np.dtype([('', handler.metadata_type) for handler in handlers],
                    align=True)
