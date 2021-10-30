
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
from ffcv.memory_managers.ram import RAMMemoryManager

from test_memory_allocation import DummyDataset


def create_and_validate(length, size, do_compile):

    dataset = DummyDataset(length, size)

    with NamedTemporaryFile() as handle:
        name = handle.name
        writer = DatasetWriter(length, name, {
            'index': IntField(),
            'value': BytesField()
        })

        with writer:
            writer.write_pytorch_dataset(dataset, num_workers=2, chunksize=5)

        reader = Reader(name)
        manager = RAMMemoryManager(reader)

        indices = np.random.choice(length, 500)
        addresses = reader.alloc_table['ptr'][indices]
        sample_ids = reader.alloc_table['sample_id'][indices]

        Compiler.set_enabled(do_compile)

        with manager:
            read_fn = manager.compile_reader()

        for addr, sample_id in zip(tqdm(addresses), sample_ids):
            read_buffer = read_fn(addr, manager.state)
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
