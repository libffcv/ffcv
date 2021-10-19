from typing import List
from os import SEEK_END, path
import numpy as np
from time import sleep
import ctypes
from multiprocessing import (shared_memory, cpu_count, Queue, Process, Value)

from tqdm import tqdm

from .utils import chunks, is_power_of_2
from .fields import Field
from .memory_allocator import MemoryAllocator
from .types import (TYPE_ID_HANDLER, get_metadata_type, HeaderType,
                    FieldDescType, CURRENT_VERSION)

MIN_PAGE_SIZE = 1 << 21  # 2MiB, which is the most common HugePage size


def worker_job(input_queue, metadata_sm, metadata_type, fields,
               allocator, done_number, dataset):

    metadata = np.frombuffer(metadata_sm.buf, dtype=metadata_type)
    field_names = metadata_type.names

    # This `with` block ensures that all the pages allocated have been written
    # onto the file
    with allocator:
        while True:
            chunk = input_queue.get()

            if chunk is None:
                # No more work left to do
                break

            # For each sample in the chunk
            for dest_ix, source_ix in chunk:
                # We extract the sample in question from the dataset
                sample = dataset[source_ix]
                # We write each field individually to the metadata region
                for field_name, field, field_value in zip(field_names, fields, sample):
                    destination = metadata[field_name][dest_ix: dest_ix + 1]
                    field.encode(destination, field_value, allocator.malloc)

            # We warn the main thread of our progress
            with done_number.get_lock():
                done_number.value += len(chunk)


class DatasetWriter():
    def __init__(self, num_samples: int, fname: str, fields: List[Field],
                 page_size: int = 4 * MIN_PAGE_SIZE):
        self.num_samples = num_samples
        self.fields = fields
        self.fname = fname
        self.metadata_type = get_metadata_type(self.fields)

        if not is_power_of_2(page_size):
            raise ValueError(f'page_size isnt a power of 2')
        if page_size < MIN_PAGE_SIZE:
            raise ValueError(f"page_size can't be lower than{MIN_PAGE_SIZE}")

        self.page_size = page_size


    def __enter__(self):
        with open(self.fname, 'wb') as fp:


            # Write the header data
            header = np.zeros(1, dtype=HeaderType)[0]
            header['version'] = CURRENT_VERSION
            header['num_samples'] = self.num_samples
            header['num_fields'] = len(self.fields)
            header['page_size'] = self.page_size
            fp.write(header.tobytes())


            # Writes the information about the fields
            fields_descriptor = np.zeros(len(self.fields),
                                              dtype=FieldDescType)
            field_type_to_type_id = {v: k for (k, v) in TYPE_ID_HANDLER.items()}
            for i, field in enumerate(self.fields):
                type_id = field_type_to_type_id[type(field)]
                fields_descriptor[i]['type_id'] = type_id
                fields_descriptor[i]['arguments'] = field.to_binary()

            fp.write(fields_descriptor.tobytes())


        # Makes a memmap to the metadata for the samples

        total_metadata_size = self.num_samples * self.metadata_type.itemsize

        # Shared memory for all the writers to fill the information
        self.metadata_sm = 3
        self.metadata_start = (HeaderType.itemsize + fields_descriptor.nbytes)

        self.metadata_sm = shared_memory.SharedMemory(create=True,
                                                      size=total_metadata_size)

        self.data_region_start = self.metadata_start + total_metadata_size



    def write_pytorch_dataset(self, dataset, num_workers=-1,
                              order: List[int]=None, chunksize=1000):

        # We use all cores by default
        if num_workers == -1:
            num_workers = cpu_count()

        # If the user didn't specify an order we just add samples
        # sequentially
        if order is None:
            order = np.arange(self.num_samples)

        # We add indices to the order so that workers know where
        # to write in the metadata array
        order = list(enumerate(order))

        # We publish all the work that has to be done into a queue
        workqueue = Queue()
        for chunk in chunks(order, chunksize):
            workqueue.put(chunk)

        # We add a token for each worker to warn them that there
        # is no more work to be done
        for _ in range(num_workers):
            workqueue.put(None)

        # Define counters we need to orchestrate the workers
        done_number = Value(ctypes.c_uint64, 0)
        allocator = MemoryAllocator(self.fname,
                                    self.data_region_start,
                                    self.page_size)

        # Arguments that have to be passed to the workers
        worker_args = (workqueue, self.metadata_sm,
                       self.metadata_type, self.fields,
                       allocator, done_number,
                       dataset)

        # Create the workers
        processes = [Process(target=worker_job, args=worker_args)
                     for _ in range(num_workers)]
        # start the workers
        [p.start() for p in processes]
        # Wait for all the workers to be done

        # Display progress
        progress = tqdm(total=self.num_samples)
        previous = 0
        while previous != self.num_samples:
            val = done_number.value
            diff = val - previous
            if diff > 0:
                progress.update(diff)
            previous = val
            sleep(0.1)
        progress.close()

        # Wait for all the workers to be done
        [p.join() for p in processes]

        # Writing metadata
        with open(self.fname, 'r+b') as fp:
            fp.seek(self.metadata_start)
            print(fp.tell())
            fp.write(self.metadata_sm.buf)
            print(fp.tell())


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.metadata_sm.close()
        self.metadata_sm.unlink()
