import os
from tempfile import NamedTemporaryFile
from time import sleep, time
import os, psutil


import numpy as np
import pytest
from tqdm import tqdm
from assertpy import assert_that
from torch.utils.data import Dataset

from ffcv.writer import DatasetWriter
from ffcv.reader import Reader
from ffcv.fields import BytesField, IntField
from ffcv.pipeline.compiler import Compiler
from ffcv import Loader

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

def create_and_run(num_samples, size_bytes):
    handle = NamedTemporaryFile()
    with handle:
        name = handle.name
        dataset = DummyDataset(num_samples, size_bytes)
        writer = DatasetWriter(num_samples, name, {
            'index': IntField(),
            'value': BytesField()
        })
        
        Compiler.set_enabled(True)

        with writer:
            writer.write_pytorch_dataset(dataset, num_workers=-1, chunksize=100)
        total_dataset_size = num_samples * size_bytes
        # Dataset should not be in RAM
        process = psutil.Process(os.getpid())
        assert_that(process.memory_info().rss).is_less_than(total_dataset_size)
        
        loader = Loader(name, 128, 10)
        for _ in tqdm(loader):
            assert_that(process.memory_info().rss).is_less_than(total_dataset_size)



@pytest.mark.skipif(bool(os.environ.get('FFCV_RUN_MEMORY_LEAK_TEST', "0")),
                    reason="set FFCV_RUN_MEMORY_LEAK_TEST to enable it")
def test_memory_leak_write():
    create_and_run(128100, 500*300*3)