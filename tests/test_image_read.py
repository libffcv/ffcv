import numpy as np
from tqdm import tqdm
from assertpy import assert_that
from torch.utils.data import Dataset
import logging
from time import time
import os
from assertpy import assert_that
from tempfile import NamedTemporaryFile

from ffcv.writer import DatasetWriter
from ffcv.reader import Reader
from ffcv.fields import IntField, RGBImageField
from ffcv.pipeline.compiler import Compiler
from ffcv.memory_managers.ram import RAMMemoryManager

class DummyDataset(Dataset):

    def __init__(self, length):
        self.length = length
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError
        
        np.random.seed(37 + index)
        dims = tuple([128, 128, 3])
        image_data = np.random.randint(low=0, high=255, size=dims, dtype='uint8')
        return index, image_data



def create_and_validate(length, mode='raw'):

    dataset = DummyDataset(length)

    with NamedTemporaryFile() as handle:
        name = handle.name
        writer = DatasetWriter(length, name, {
            'index': IntField(),
            'value': RGBImageField(write_mode=mode)
        })

        with writer:
            writer.write_pytorch_dataset(dataset, num_workers=2, chunksize=5)
            
        reader = Reader(name)
        manager = RAMMemoryManager(reader)
        with manager:
            Decoder = RGBImageField().get_decoder_class()
            decoder = Decoder()
            decoder.accept_globals(reader.metadata['f1'], manager.compile_reader())

        decode = decoder.generate_code()

        assert_that(reader.metadata).is_length(length)
        buff = np.zeros((1, 128, 128, 3), dtype='uint8')
        
        for i in range(length):
            result = decode(np.array([i]), buff)[0]
            _, ref_image = dataset[i]
            assert_that(result.shape).is_equal_to(ref_image.shape)
            if mode == 'jpg':
                dist = np.abs(ref_image.astype('float') - result.astype('float'))
                assert_that(dist.mean()).is_less_than(80)
            else:
                assert_that(np.all(ref_image == result)).is_true()
                
def test_simple_image_dataset_raw():
    create_and_validate(500, 'raw')

def test_simple_image_dataset_jpg():
    create_and_validate(100, 'jpg')
