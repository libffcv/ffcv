from ctypes import pointer
from tempfile import NamedTemporaryFile
from collections import defaultdict
from assertpy.assertpy import assert_that
from multiprocessing import cpu_count

import torch as ch
from assertpy import assert_that
import numpy as np
from torch.utils.data import Dataset
from ffcv import DatasetWriter
from ffcv.fields import IntField, NDArrayField, TorchTensorField
from ffcv import Loader

class DummyActivationsDataset(Dataset):

    def __init__(self, n_samples, shape, is_ch=False):
        self.n_samples = n_samples
        self.shape = shape
        self.is_ch = is_ch

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if index >= self.n_samples:
            raise IndexError()
        np.random.seed(index)
        to_return = np.random.randn(*self.shape).astype('<f4')
        if self.is_ch:
            to_return = ch.from_numpy(to_return)

        return index, to_return

class TripleDummyActivationsDataset(Dataset):

    def __init__(self, n_samples, shape):
        self.n_samples = n_samples
        self.shape = shape

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if index >= self.n_samples:
            raise IndexError()
        np.random.seed(index)
        d1 = np.random.randn(*self.shape).astype('<f4')
        d2 = np.random.randn(*self.shape).astype('<f4')
        d3 = np.random.randn(*self.shape).astype('<f4')
        return index, d1, d2, d3

def run_test(n_samples, shape, is_ch=False):
    with NamedTemporaryFile() as handle:
        name = handle.name
        dataset = DummyActivationsDataset(n_samples, shape, is_ch)

        if is_ch:
            field = TorchTensorField(ch.float32, shape)
        else:
            field = NDArrayField(np.dtype('<f4'), shape)

        writer = DatasetWriter(name, {
            'index': IntField(),
            'activations': field
        }, num_workers=3)

        writer.from_indexed_dataset(dataset)

        loader = Loader(name, batch_size=3, num_workers=min(5, cpu_count()))
        for ixes, activations in loader:
            for ix, activation in zip(ixes, activations):
                d = dataset[ix][1]
                if is_ch:
                    d = d.numpy()
                assert_that(np.all(d == activation.numpy())).is_true()


def test_simple_activations():
    run_test(4096, (2048,))

def test_simple_activations_ch():
    run_test(4096, (2048,), True)

def test_multi_fields():
    n_samples = 4096
    shape = (2048,)

    with NamedTemporaryFile() as handle:
        name = handle.name
        dataset = TripleDummyActivationsDataset(n_samples, shape)
        writer = DatasetWriter(name, {
            'index': IntField(),
            'activations': NDArrayField(np.dtype('<f4'), shape),
            'activations2': NDArrayField(np.dtype('<f4'), shape),
            'activations3': NDArrayField(np.dtype('<f4'), shape)
        }, num_workers=1)


        writer.from_indexed_dataset(dataset)

        loader = Loader(name, batch_size=3, num_workers=min(5, cpu_count()))
        page_size_l2 = int(np.log2(loader.reader.page_size))
        sample_ids = loader.reader.alloc_table['sample_id']
        pointers = loader.reader.alloc_table['ptr']
        pages = pointers >> page_size_l2
        sample_to_pages = defaultdict(set)

        for sample_id, page in zip(sample_ids, pages):
            sample_to_pages[sample_id].add(page)
            assert_that(sample_to_pages[sample_id]).is_length(1)

        for ixes, activations, d2, d3 in loader:
            for ix, activation in zip(ixes, activations):
                assert_that(np.all(dataset[ix][1] == activation.numpy())).is_true()