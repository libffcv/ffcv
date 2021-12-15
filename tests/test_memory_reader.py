
import numpy as np
from tqdm import tqdm
from assertpy import assert_that
from torch.utils.data import Dataset
import logging
from time import time
import os
from assertpy import assert_that
from tempfile import NamedTemporaryFile

from ffcv.writer import DatasetWriter
from ffcv.reader import Reader
from ffcv.fields import BytesField, IntField
from ffcv.pipeline.compiler import Compiler
from ffcv.memory_managers import OSCacheManager

from test_memory_allocation import DummyDataset


def create_and_validate(length, size, do_compile):

    dataset = DummyDataset(length, size)

    with NamedTemporaryFile() as handle:
        name = handle.name
        writer = DatasetWriter(name, {
            'index': IntField(),
            'value': BytesField()
        }, num_workers=2)

        writer.from_indexed_dataset(dataset, chunksize=5)

        reader = Reader(name)
        manager = OSCacheManager(reader)
        context = manager.schedule_epoch(np.array([0, 1]))

        indices = np.random.choice(length, 500)
        addresses = reader.alloc_table['ptr'][indices]
        sample_ids = reader.alloc_table['sample_id'][indices]

        Compiler.set_enabled(do_compile)
        read_fn = manager.compile_reader()

        with context:

            for addr, sample_id in zip(tqdm(addresses), sample_ids):
                read_buffer = read_fn(addr, context.state)
                np.random.seed(sample_id)
                expected_buff = np.random.randint(0, 255, size=size, dtype='u1')

                assert_that(read_buffer).is_length(len(expected_buff))
                assert_that(np.all(read_buffer == expected_buff)).is_true()

            # We skip the first which is compilation

def test_simple():
    create_and_validate(600, 76, False)

def test_large():
    create_and_validate(600, 1024, False)

def test_many():
    create_and_validate(60000, 81, False)

def test_many_compiled():
    create_and_validate(1000000, 1, True)
