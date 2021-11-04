from collections import defaultdict
from tempfile import TemporaryDirectory
from os import path
from typing import Counter

from assertpy import assert_that
import numpy as np
from torch.utils.data import Dataset, distributed
from torch.multiprocessing import spawn, Queue
from torch.distributed import init_process_group

from ffcv.loader.loader import ORDER_TYPE, OrderOption
from ffcv.fields.basics import FloatField
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, FloatField
from ffcv import Loader

class DummyDataset(Dataset):

    def __init__(self, l):
        self.l = l

    def __len__(self):
        return self.l

    def __getitem__(self, index):
        if index > self.l:
            raise IndexError()
        return (index, np.sin(index))


def process_work(rank, world_size, fname, order, sync_fname, out_folder):
    sync_url = f'file://{sync_fname}'
    init_process_group('nccl', sync_url, rank=rank, world_size=world_size)
    
    loader = Loader(fname, 8, num_workers=2, order=order,
                    distributed=world_size > 1)
    
    result = []
    for _ in range(3):
        content = np.concatenate([x[0].numpy().reshape(-1).copy() for x in loader])
        result.append(content)
    result = np.stack(result)
    np.save(path.join(out_folder, f"result-{rank}.npy"), result)
    

def prep_and_run_test(num_workers, order):
    length = 600
    with TemporaryDirectory() as folder:
            name = path.join(folder, 'dataset.beton')
            sync_file = path.join(folder, 'share')
            dataset = DummyDataset(length)
            writer = DatasetWriter(length, name, {
                'index': IntField(),
                'value': FloatField()
            })

            with writer:
                writer.write_pytorch_dataset(dataset)
                
            spawn(process_work, nprocs=num_workers, args=(num_workers,
                                                          name,
                                                          order,
                                                          sync_file,
                                                          folder))
            
            results = []
            for r in range(num_workers):
                array = np.load(path.join(folder,f"result-{r}.npy"))
                results.append(array)

            results = np.concatenate(results, 1)
            
            for i in range(results.shape[0]):
                if order == OrderOption.SEQUENTIAL and i < results.shape[0] - 1:
                    assert_that((results[i] == results[i + 1]).all()).is_true()
                if order == OrderOption.RANDOM and i < results.shape[0] - 1:
                    assert_that((results[i] == results[i + 1]).all()).is_false()
                    
                epoch_content = Counter(results[i])
                indices_gotten = np.array(sorted(list(epoch_content.keys())))
                assert_that(np.all(np.arange(length) == indices_gotten)).is_true()
                assert_that(min(epoch_content.values())).is_equal_to(1)
                assert_that(max(epoch_content.values())).is_less_than_or_equal_to(2)
        
        

def test_distributed_sequential_1():
    prep_and_run_test(1, OrderOption.SEQUENTIAL)

def test_distributed_sequential_2():
    prep_and_run_test(2, OrderOption.SEQUENTIAL)

def test_distributed_sequential_3():
    prep_and_run_test(3, OrderOption.SEQUENTIAL)

def test_distributed_sequential_4():
    prep_and_run_test(4, OrderOption.SEQUENTIAL)

def test_distributed_random_1():
    prep_and_run_test(1, OrderOption.RANDOM)

def test_distributed_random_2():
    prep_and_run_test(2, OrderOption.RANDOM)

def test_distributed_random_3():
    prep_and_run_test(3, OrderOption.RANDOM)

def test_distributed_random_4():
    prep_and_run_test(4, OrderOption.RANDOM)