import numpy as np
from assertpy import assert_that
from torch.utils.data import Dataset
import logging
import os
from assertpy import assert_that
from tempfile import NamedTemporaryFile

from ffcv.writer import DatasetWriter
from ffcv.reader import Reader
from ffcv.fields import IntField, FloatField, BytesField

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

class DummyDataset(Dataset):

    def __init__(self, l):
        self.l = l

    def __len__(self):
        return self.l

    def __getitem__(self, index):
        if index > self.l:
            raise IndexError()
        return (index, np.sin(index))

class DummyDatasetWithData(Dataset):

    def __init__(self, l):
        self.l = l

    def __len__(self):
        return self.l

    def __getitem__(self, index):
        if index > self.l:
            raise IndexError()
        return (index, np.zeros(2))

def validate_simple_dataset(name, length):
    reader = Reader(name)
    assert_that(reader.handlers).is_length(2)
    assert_that(reader.handlers['index']).is_instance_of(IntField)
    assert_that(reader.handlers['value']).is_instance_of(FloatField)
    assert_that(reader.alloc_table).is_length(0)
    assert_that(reader.metadata).is_length(length)
    assert_that((reader.metadata['f0'] == np.arange(length).astype('int')).all()).is_true()
    assert_that((np.sin(reader.metadata['f0']) == reader.metadata['f1']).all()).is_true()

def test_write_simple():
    length = 600
    with NamedTemporaryFile() as handle:
        name = handle.name
        dataset = DummyDataset(length)
        writer = DatasetWriter(length, name, {
            'index': IntField(),
            'value': FloatField()
        })

        with writer:
            writer.write_pytorch_dataset(dataset)

        validate_simple_dataset(name, length)

def test_multiple_workers():
    length = 600
    with NamedTemporaryFile() as handle:
        name = handle.name
        dataset = DummyDataset(length)
        writer = DatasetWriter(length, name, {
            'index': IntField(),
            'value': FloatField()
        })

        with writer:
            writer.write_pytorch_dataset(dataset, num_workers=30, chunksize=10000)

        validate_simple_dataset(name, length)


def test_super_long():
    length = 600000
    with NamedTemporaryFile() as handle:
        name = handle.name
        dataset = DummyDataset(length)
        writer = DatasetWriter(length, name, {
            'index': IntField(),
            'value': FloatField()
        })

        with writer:
            writer.write_pytorch_dataset(dataset, num_workers=30, chunksize=10000)

        validate_simple_dataset(name, length)

def test_small_chunks_multiple_workers():
    length = 600
    with NamedTemporaryFile() as handle:
        name = handle.name
        dataset = DummyDatasetWithData(length)
        writer = DatasetWriter(length, name, {
            'index': IntField(),
            'value': BytesField()
        })

        with writer:
            writer.write_pytorch_dataset(dataset, num_workers=30, chunksize=1)