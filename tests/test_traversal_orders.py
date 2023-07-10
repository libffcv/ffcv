from tempfile import TemporaryDirectory
from os import path
from typing import Counter

import pytest
from assertpy import assert_that
import numpy as np
from torch.utils.data import Dataset
from torch.multiprocessing import spawn
from torch.distributed import init_process_group

from ffcv.loader.loader import ORDER_TYPE, OrderOption
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, BytesField
from ffcv import Loader

class DummyDataset(Dataset):

    def __init__(self, l):
        self.l = l

    def __len__(self):
        return self.l

    def __getitem__(self, index):
        if index > self.l:
            raise IndexError()
        return (index, np.sin(np.array([index])).view('<u1'))


def process_work(rank, world_size, fname, order, sync_fname, out_folder, indices):
    sync_url = f'file://{sync_fname}'
    if world_size > 1:
        init_process_group('nccl', sync_url, rank=rank, world_size=world_size)
    
    loader = Loader(fname, 8, num_workers=2, order=order, drop_last=False,
                    distributed=world_size > 1, indices=indices)
    
    result = []
    for _ in range(3):
        content = np.concatenate([x[0].numpy().reshape(-1).copy() for x in loader])
        result.append(content)
    result = np.stack(result)
    
    np.save(path.join(out_folder, f"result-{rank}.npy"), result)
    

def prep_and_run_test(num_workers, order, with_indices=False):
    length = 600
    indices = None
    if with_indices:
        indices = np.random.choice(length, length//2, replace=False)

    with TemporaryDirectory() as folder:
            name = path.join(folder, 'dataset.beton')
            sync_file = path.join(folder, 'share')
            dataset = DummyDataset(length)
            writer = DatasetWriter(name, {
                'index': IntField(),
                'value': BytesField()
            })

            writer.from_indexed_dataset(dataset)
                
            args = (num_workers, name, order, sync_file, folder, indices)
            if num_workers > 1:
                spawn(process_work, nprocs=num_workers, args=args)
            else:
                process_work(*((0, ) + args))
            
            results = []
            for r in range(num_workers):
                array = np.load(path.join(folder,f"result-{r}.npy"))
                results.append(array)

            results = np.concatenate(results, 1)
            
            # For each epoch
            for i in range(results.shape[0]):
                if not with_indices:
                    if order == OrderOption.SEQUENTIAL and i < results.shape[0] - 1:
                        assert_that((results[i] == results[i + 1]).all()).is_true()
                    if order != OrderOption.SEQUENTIAL and i < results.shape[0] - 1:
                        assert_that((results[i] == results[i + 1]).all()).is_false()
                        
                    epoch_content = Counter(results[i])
                    indices_gotten = np.array(sorted(list(epoch_content.keys())))
                    assert_that(np.all(np.arange(length) == indices_gotten)).is_true()
                    assert_that(min(epoch_content.values())).is_equal_to(1)
                    assert_that(max(epoch_content.values())).is_less_than_or_equal_to(2)
                else:
                    assert_that(set(results[i])).is_equal_to(set(indices))
                

def test_traversal_sequential_1():
    prep_and_run_test(1, OrderOption.SEQUENTIAL)

def test_traversal_sequential_2():
    prep_and_run_test(2, OrderOption.SEQUENTIAL)

def test_traversal_sequential_3():
    prep_and_run_test(3, OrderOption.SEQUENTIAL)

def test_traversal_sequential_4():
    prep_and_run_test(4, OrderOption.SEQUENTIAL)

def test_traversal_random_1():
    prep_and_run_test(1, OrderOption.RANDOM)

def test_traversal_random_2():
    prep_and_run_test(2, OrderOption.RANDOM)

def test_traversal_random_3():
    prep_and_run_test(3, OrderOption.RANDOM)

def test_traversal_random_4():
    prep_and_run_test(4, OrderOption.RANDOM)

def test_traversal_quasirandom_1():
    prep_and_run_test(1, OrderOption.QUASI_RANDOM)

def test_traversal_quasirandom_2():
    prep_and_run_test(2, OrderOption.QUASI_RANDOM)

def test_traversal_quasirandom_3():
    prep_and_run_test(3, OrderOption.QUASI_RANDOM)

def test_traversal_quasirandom_4():
    prep_and_run_test(4, OrderOption.QUASI_RANDOM)

def test_traversal_sequential_distributed_with_indices():
    prep_and_run_test(2, OrderOption.SEQUENTIAL, True)

def test_traversal_random_distributed_with_indices():
    prep_and_run_test(2, OrderOption.RANDOM, True)

def test_traversal_quasi_random_distributed_with_indices():
    prep_and_run_test(2, OrderOption.QUASI_RANDOM, True)
