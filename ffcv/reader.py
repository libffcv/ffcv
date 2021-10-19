import numpy as np
from .types import (HeaderType, CURRENT_VERSION, FieldDescType, get_handlers, get_metadata_type)
from .memory_allocator import align_to_page

class Reader:

    def __init__(self, fname):
        self._fname = fname
        self.read_header()
        self.read_field_descriptors()
        self.read_metadata()

    def read_header(self):
        header = np.fromfile(self._fname, dtype=HeaderType, count=1)[0]
        header.setflags(write=False)
        version = header['version']

        if version != CURRENT_VERSION:
            msg = f"file format mismatch: code={CURRENT_VERSION},file={version}"
            raise AssertionError(msg)

        self.num_samples = header['num_samples']
        self.page_size = header['page_size']
        self.num_fields = header['num_fields']
        self.header = header

    def read_field_descriptors(self):
        offset = HeaderType.itemsize
        field_descriptors = np.fromfile(self._fname, dtype=FieldDescType,
                                        count=self.num_fields, offset=offset)
        field_descriptors.setflags(write=False)
        handlers = get_handlers(field_descriptors)

        self.field_descriptors = field_descriptors
        self.metadata_type = get_metadata_type(handlers)
        self.handlers = handlers

    def read_metadata(self):
        offset = HeaderType.itemsize + self.field_descriptors.nbytes
        self.metadta = np.fromfile(self._fname, dtype=self.metadata_type,
                                   count=self.num_samples, offset=offset)
        self.metadta.setflags(write=False)

