from dataclasses import replace
import numpy as np
from typing import Callable
from assertpy import assert_that
from torch.utils.data import Dataset
import logging
import os
from assertpy import assert_that
from tempfile import NamedTemporaryFile
from ffcv.pipeline.operation import Operation
from ffcv.transforms.ops import ToTensor

from ffcv.writer import DatasetWriter
from ffcv.reader import Reader
from ffcv.loader import Loader
from ffcv.fields import IntField, FloatField, BytesField
from ffcv.fields.basics import FloatDecoder
from ffcv.pipeline.state import State

from test_writer import DummyDataset

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)



def validate_simple_dataset(name, length):
    reader = Reader(name)
    assert_that(reader.handlers).is_length(2)
    assert_that(reader.handlers['index']).is_instance_of(IntField)
    assert_that(reader.handlers['value']).is_instance_of(FloatField)
    assert_that(reader.alloc_table).is_length(0)
    assert_that(reader.metadata).is_length(length)
    assert_that((reader.metadata['f0'] == np.arange(length).astype('int')).all()).is_true()
    assert_that((np.sin(reader.metadata['f0']) == reader.metadata['f1']).all()).is_true()

class Doubler(Operation):

    def generate_code(self) -> Callable:
        def code(x, dst):
            dst[:] = x * 2
        return code

    def declare_state_and_memory(self, previous_state: State):
        return (previous_state, None)

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

        loader = Loader(name, 128, 5, seed=17,
        pipelines={
            'value': [FloatDecoder(), Doubler(), ToTensor()]
        })
        it = iter(loader)
        
        it.generate_code()