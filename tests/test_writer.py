import numpy as np
from assertpy import assert_that
from torch.utils.data import Dataset
import logging
import os
from assertpy import assert_that
from tempfile import NamedTemporaryFile

from ffcv.writer import DatasetWriter
from ffcv.reader import Reader
from ffcv.fields import IntField, FloatField

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

def test_write_simple():
    length = 600
    with NamedTemporaryFile() as handle:
        name = handle.name
        dataset = DummyDataset(length)
        writer = DatasetWriter(600, name, {
            'index': IntField(),
            'value': FloatField()
        })

        with writer:
            writer.write_pytorch_dataset(dataset)

        reader = Reader(name)

        assert_that(reader.handlers).is_length(2)
        assert_that(reader.handlers['index']).is_instance_of(IntField)
        assert_that(reader.handlers['value']).is_instance_of(FloatField)
        assert_that(reader.alloc_table).is_length(0)
        assert_that(reader.metadata).is_length(length)
        assert_that((reader.metadata['f0'] == np.arange(length).astype('int')).all()).is_true()
        assert_that((np.sin(reader.metadata['f0']) == reader.metadata['f1']).all()).is_true()

        print(reader.metadata)

