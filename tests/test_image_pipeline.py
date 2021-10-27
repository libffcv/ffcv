import numpy as np
import torch as ch
from torch.utils.data import Dataset
from assertpy import assert_that
from tempfile import NamedTemporaryFile

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from ffcv.loader import Loader
from ffcv.pipeline.compiler import Compiler

class DummyDataset(Dataset):

    def __init__(self, length, height, width):
        self.length = length
        self.height = height
        self.width = width
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index > self.length:
            raise IndexError
        dims = (self.height, self.width, 3)
        image_data = ((np.ones(dims) * index) % 255).astype('uint8')
        return index, image_data

def create_and_validate(length, mode='raw'):

    dataset = DummyDataset(length, 5, 6)

    with NamedTemporaryFile() as handle:
        name = handle.name
        writer = DatasetWriter(length, name, {
            'index': IntField(),
            'value': RGBImageField(write_mode=mode)
        })

        with writer:
            writer.write_pytorch_dataset(dataset, num_workers=2, chunksize=5)
            
            
        Compiler.set_enabled(False)
        
        loader = Loader(name, batch_size=5, num_workers=2)
        
        for index, images in loader:
            for i, image in zip(index, images):
                assert_that(ch.all((image == (i % 255)).reshape(-1))).is_true()
                
def test_simple_raw_image_pipeline():
    create_and_validate(500, 'raw')
