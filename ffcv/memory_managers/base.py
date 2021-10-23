from abc import abstractmethod, ABCMeta
from contextlib import AbstractContextManager
from collections import defaultdict
from typing import Mapping, Set

import numpy as np
from numba.typed import Dict
from numba import types

from ..reader import Reader
from ..pipeline.compiler import Compiler

class MemoryManager(AbstractContextManager, metaclass=ABCMeta):

    def __init__(self, reader:Reader):
        self.reader = reader
        alloc_table = self.reader.alloc_table

        # Table mapping any address in the file to the size of the data region
        # That was allocated there
        self.ptrs = alloc_table['ptr']
        self.sizes = alloc_table['size']
        order = np.argsort(self.ptrs)
        # Order them so that we can use search sorted
        self.ptrs = self.ptrs[order]
        self.sizes =  self.sizes[order]

        self.ptr_to_size = dict(zip(self.ptrs, self.sizes))

        # We extract the page number by shifting the address corresponding
        # to the page width
        page_size_bit_location = int(np.log2(reader.page_size))
        page_locations = alloc_table['ptr'] >> page_size_bit_location

        sample_to_pages: Mapping[int, Set[int]] = defaultdict(set)
        page_to_samples: Mapping[int, Set[int]] = defaultdict(set)

        # We create a mapping that goes from sample id to the pages it has data
        # Stored to
        # (And the same for the other way around)
        for sid, pid in zip(alloc_table['sample_id'], page_locations):
            sample_to_pages[sid].add(pid)
            page_to_samples[pid].add(sid)

        self.sample_to_pages = sample_to_pages
        self.page_to_samples = page_to_samples

        super().__init__()

    @abstractmethod
    def schedule_epoch(self, schedule):
        raise NotImplemented()

    @abstractmethod
    def compile_reader(self, address, size):
        raise NotImplemented()

    @abstractmethod
    def __enter__(self):
        return super().__enter__()

    @abstractmethod
    def __exit__(self, __exc_type, __exc_value, __traceback):
        return super().__exit__(__exc_type, __exc_value, __traceback)
