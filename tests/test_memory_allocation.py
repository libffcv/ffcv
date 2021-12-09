import numpy as np
from assertpy import assert_that
from torch.utils.data import Dataset
import logging
import os
from assertpy import assert_that
from tempfile import NamedTemporaryFile

from ffcv.writer import DatasetWriter
from ffcv.reader import Reader
from ffcv.fields import BytesField, IntField

class DummyDataset(Dataset):

    def __init__(self, l, size):
        self.l = l
        self.size = size

    def __len__(self):
        return self.l

    def __getitem__(self, index):
        if index > self.l:
            raise IndexError
        np.random.seed(index)
        return index, np.random.randint(0, 255, size=self.size, dtype='u1')


def create_and_validate(length, size):

    dataset = DummyDataset(length, size)

    with NamedTemporaryFile() as handle:
        name = handle.name
        writer = DatasetWriter(name, {
            'index': IntField(),
            'value': BytesField()
        }, num_workers=2)

        writer.from_indexed_dataset(dataset, chunksize=5)

        reader = Reader(name)

        assert_that(reader.handlers).is_length(2)
        assert_that(reader.handlers['index']).is_instance_of(IntField)
        assert_that(reader.handlers['value']).is_instance_of(BytesField)
        assert_that(reader.alloc_table).is_length(length)
        assert_that(reader.metadata).is_length(length)
        assert_that((reader.metadata['f0'] == np.arange(length).astype('int')).all()).is_true()

        assert_that(np.all(reader.alloc_table['size'] == size)).is_true()

def test_simple():
    create_and_validate(600, 76)

def test_large():
    create_and_validate(600, 1024)

def test_many():
    create_and_validate(60000, 81)


