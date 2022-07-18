import numpy as np
from time import sleep
from os import SEEK_END
from multiprocessing import Value
from .utils import align_to_page
import ctypes

class MemoryAllocator():
    def __init__(self, fname, offset_start, page_size):
        self.fname = fname
        self.offset = align_to_page(offset_start, page_size)
        self.next_page_allocated  = Value(ctypes.c_uint64, 0)
        self.next_page_written  = Value(ctypes.c_uint64, 0)

        self.page_size = page_size
        self.page_offset = 0
        self.my_page = -1

        self.page_data = np.zeros(self.page_size, '<u1')
        self.allocations = []
        self.current_sample_id = None

    def __enter__(self):
        self.fp = open(self.fname, 'ab', buffering=0)

    def set_current_sample(self, current_sample_id):
        self.current_sample_id = current_sample_id

    @property
    def space_left_in_page(self):
        # We don't have a page allocated yet
        if self.my_page < 0:
            return 0
        return self.page_size - self.page_offset

    def malloc(self, size):
        # print(f"Allocating {size} bytes")
        if size > self.page_size:
            raise ValueError(f"Tried allocating {size} but" +
                             f" page size is {self.page_size}")

        if size > self.space_left_in_page:
            self.flush_page()
            # We book the next available page in the file
            with self.next_page_allocated.get_lock():
                self.my_page = self.next_page_allocated.value
                self.next_page_allocated.value = self.my_page + 1

            self.page_offset = 0
            # This is a new page so we erate the content of the buffer
            self.page_data.fill(0)

            # We check if we already allocated space for this sample on
            # the page that is now full
            region_in_previous_page = False
            while self.allocations and self.allocations[-1][0] == self.current_sample_id:
                # We have to revert the allocations we did and we are giving
                # up on this sample.
                self.allocations.pop()
                # We found at least memory region from the preivous page
                region_in_previous_page = True

            # The writer will restart from this freshly allocated page
            if region_in_previous_page:
                raise MemoryError("Not enough memory to fit the whole sample")

        previous_offset = self.page_offset
        self.page_offset += size

        buffer = self.page_data[previous_offset:self.page_offset]
        ptr = self.offset + self.my_page * self.page_size + previous_offset

        # We return the pointer to the location in file and where to write
        # the data
        self.allocations.append((self.current_sample_id, ptr, size))
        return ptr, buffer

    def flush_page(self):
        # If we haven't allocated any page we end there
        if self.my_page < 0:
            return

        # We shouldn't have allocated a page and have nothing to write on it
        assert self.page_offset != 0
        # Wait until it's my turn to write
        while self.next_page_written.value != self.my_page:
            # Essentially a spin lock
            # TODO we could replace it with like exponential backoff
            sleep(0.001)
            pass

        # Now it's my turn to write

        expected_file_offset = self.offset + self.my_page * self.page_size
        # in order to be aligned with page size
        # If this is the first page we have to pad with zeros
        if self.my_page == 0:
            # print("Padding headers to align with page size")
            current_location = self.fp.seek(0, SEEK_END)
            null_bytes_to_write = expected_file_offset - current_location
            self.fp.write(np.zeros(null_bytes_to_write, dtype='<u1').tobytes())
            # print(f"current file pointer is no {self.fp.tell()} and should be {expected_file_offset}")


        self.fp.seek(expected_file_offset)

        # print(f"Writing page {self.my_page} at offset {self.fp.tell()}")
        self.fp.write(self.page_data.tobytes())
        # print(f"Done writing {self.my_page} at offset {self.fp.tell()}")

        # We warn other processes that they are free to write the next page
        with self.next_page_written.get_lock():
            self.next_page_written.value += 1



    # Flush the last page and
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush_page()
        self.fp.close()
