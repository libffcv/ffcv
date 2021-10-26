import logging
import os
from tempfile import NamedTemporaryFile
from time import sleep, time

import numpy as np
from assertpy import assert_that
from ffcv.fields import BytesField, IntField, RGBImageField
from ffcv.memory_managers.ram import RAMMemoryManager
from ffcv.pipeline.compiler import Compiler
from ffcv.reader import Reader
from ffcv.writer import DatasetWriter
from torch.utils.data import Dataset
from tqdm import tqdm

from ..benchmark import Benchmark
from ..decorator import benchmark



class DummyDataset(Dataset):

    def __init__(self, length, min_size, max_size):
        self.length = length
        self.min_size = min_size
        self.max_size = max_size
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index > self.length:
            raise IndexError
        
        np.random.seed(37 + index)
        dims = np.random.randint(low=self.min_size, high=self.max_size,
                                 size=(2,))
        dims = tuple([*dims, 3])
        image_data = np.random.randint(low=0, high=255, size=dims, dtype='uint8')
        return index, image_data



@benchmark({
    'n': [3000],
    'length': [30000],
    'mode': [
        # 'raw',
        'jpg'
        ],
    'size_range': [
        # (32, 33),  # CIFAR
        (300, 500),  # ImageNet
    ],
    'random_reads': [
        True,
        # False
    ]
})
class ImageReadBench(Benchmark):
    
    def __init__(self, n, length, mode, size_range, random_reads):
        self.n = n
        self.mode = mode
        self.length = length
        self.size_range = size_range
        self.random_reads = random_reads
        self.dataset = DummyDataset(length, size_range[0], size_range[1])
        
    def __enter__(self):
        self.handle = NamedTemporaryFile()
        self.handle.__enter__()
        name = self.handle.name

        writer = DatasetWriter(self.length, name, {
            'index': IntField(),
            'value': RGBImageField(write_mode=self.mode)
        })

        with writer:
            writer.write_pytorch_dataset(self.dataset, num_workers=-1, chunksize=100)

        reader = Reader(name)
        manager = RAMMemoryManager(reader)

        Compiler.set_enabled(True)

        with manager:
            memreader = manager.compile_reader()
            Decoder = RGBImageField().get_decoder_class()
            decoder = Decoder()
            decoder.accept_globals(reader.metadata, memreader)

        decode = decoder.generate_code()
        decode = Compiler.compile(decode)
        

        self.buff = np.zeros((500, 500, 3), dtype='uint8')
        
        if self.random_reads:
            self.indices = np.random.choice(self.n, size=self.n, replace=False)
        else:
            self.indices = np.arange(self.n)
            
        def code(indices, buff):
            result = 0
            for i in indices:
                result += decode(reader.metadata['f1'][i], buff)[5, 5]
            return result
                
        self.code = code

    def run(self):
        self.code(self.indices, self.buff)

    def __exit__(self, *args):
        self.handle.__exit__(*args)
        pass
