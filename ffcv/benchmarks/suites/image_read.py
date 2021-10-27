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

    def __init__(self, length, size):
        self.length = length
        self.size = size
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index > self.length:
            raise IndexError

        dims = tuple([*self.size, 3])
        image_data = np.random.randint(low=0, high=255, size=dims, dtype='uint8')
        return index, image_data



@benchmark({
    'n': [3000],
    'length': [3000],
    'mode': [
        'raw',
        # 'jpg'
        ],
    'size': [
        # (32, 32),  # CIFAR
        (300, 500),  # ImageNet
    ],
    'compile': [
        True,
        False
    ],
    'random_reads': [
        True,
        # False
    ]
})
class ImageReadBench(Benchmark):
    
    def __init__(self, n, length, mode, size, random_reads, compile):
        self.n = n
        self.mode = mode
        self.length = length
        self.size = size
        self.compile = compile
        self.random_reads = random_reads
        self.dataset = DummyDataset(length, size)
        
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

        Compiler.set_enabled(self.compile)

        with manager:
            memreader = manager.compile_reader()
            Decoder = RGBImageField().get_decoder_class()
            decoder = Decoder()
            decoder.accept_globals(reader.metadata['f1'], memreader)

        decode = decoder.generate_code()
        decode = Compiler.compile(decode)
        

        self.buff = np.zeros((1, *self.size, 3), dtype='uint8')
        
        if self.random_reads:
            self.indices = np.random.choice(self.n, size=self.n, replace=False)
        else:
            self.indices = np.arange(self.n)
            
        def code(indices, buff):
            result = 0
            for i in range(len(indices)):
                result += decode(indices[i:i+1], buff)[0, 5, 5]
            return result
                
        self.code = code

    def run(self):
        self.code(self.indices, self.buff)

    def __exit__(self, *args):
        self.handle.__exit__(*args)
        pass
