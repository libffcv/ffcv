from dataclasses import replace
import torch as ch
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.compiler import Compiler
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
from multiprocessing import cpu_count

from ffcv.writer import DatasetWriter
from ffcv.reader import Reader
from ffcv.loader import Loader
from ffcv.fields import IntField, FloatField, BytesField
from ffcv.fields.basics import FloatDecoder
from ffcv.pipeline.state import State

from test_writer import DummyDataset

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


class Doubler(Operation):

    def generate_code(self) -> Callable:
        def code(x, dst):
            dst[:] = x * 2
            return dst
        return code

    def declare_state_and_memory(self, previous_state: State):
        return (previous_state, AllocationQuery(previous_state.shape, previous_state.dtype, previous_state.device))

def test_basic_simple():
    length = 600
    batch_size = 8
    with NamedTemporaryFile() as handle:
        file_name = handle.name
        dataset = DummyDataset(length)
        writer = DatasetWriter(file_name, {
            'index': IntField(),
            'value': FloatField()
        })

        writer.from_indexed_dataset(dataset)

        Compiler.set_enabled(True)

        loader = Loader(file_name, batch_size, num_workers=min(5, cpu_count()), seed=17,
                        pipelines={
                            'value': [FloatDecoder(), Doubler(), ToTensor()]
                        })

        it = iter(loader)
        indices, values = next(it)
        assert_that(np.allclose(indices.squeeze().numpy(),
                                np.arange(batch_size))).is_true()
        assert_that(np.allclose(2 * np.sin(np.arange(batch_size)),
                                values.squeeze().numpy())).is_true()

def test_multiple_iterators_success():
    length = 60
    batch_size = 8
    with NamedTemporaryFile() as handle:
        file_name = handle.name
        dataset = DummyDataset(length)
        writer = DatasetWriter(file_name, {
            'index': IntField(),
            'value': FloatField()
        })

        writer.from_indexed_dataset(dataset)

        Compiler.set_enabled(True)

        loader = Loader(file_name, batch_size, num_workers=min(5, cpu_count()), seed=17,
                        pipelines={
                            'value': [FloatDecoder(), Doubler(), ToTensor()]
                        })

        it = iter(loader)
        it = iter(loader)

def test_multiple_epoch_doesnt_recompile():
    length = 60
    batch_size = 8
    with NamedTemporaryFile() as handle:
        file_name = handle.name
        dataset = DummyDataset(length)
        writer = DatasetWriter(file_name, {
            'index': IntField(),
            'value': FloatField()
        })

        writer.from_indexed_dataset(dataset)

        Compiler.set_enabled(True)

        loader = Loader(file_name, batch_size, num_workers=min(5, cpu_count()), seed=17,
                        pipelines={
                            'value': [FloatDecoder(), Doubler(), ToTensor()]
                        })

        it = iter(loader)
        code = loader.code
        it = iter(loader)
        new_code = loader.code
        assert_that(code).is_equal_to(new_code)

def test_multiple_epoch_does_recompile():
    length = 60
    batch_size = 8
    with NamedTemporaryFile() as handle:
        file_name = handle.name
        dataset = DummyDataset(length)
        writer = DatasetWriter(file_name, {
            'index': IntField(),
            'value': FloatField()
        })

        writer.from_indexed_dataset(dataset)

        Compiler.set_enabled(True)

        loader = Loader(file_name, batch_size, num_workers=min(5, cpu_count()), seed=17,
                recompile=True,
                        pipelines={
                            'value': [FloatDecoder(), Doubler(), ToTensor()]
                        })

        it = iter(loader)
        code = loader.code
        it = iter(loader)
        new_code = loader.code
        assert_that(code).is_not_equal_to(new_code)