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

    def __init__(self, length, height, width, reversed=False):
        self.length = length
        self.height = height
        self.width = width
        self.reversed = reversed
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index > self.length:
            raise IndexError
        dims = (self.height, self.width, 3)
        image_data = ((np.ones(dims) * index) % 255).astype('uint8')
        result = index,image_data
        if self.reversed:
            result = tuple(reversed(result))
        return result

def create_and_validate(length, mode='raw', reversed=False):

    dataset = DummyDataset(length, 500, 300, reversed=reversed)

    with NamedTemporaryFile() as handle:
        name = handle.name
        
        fields = {
            'index': IntField(),
            'value': RGBImageField(write_mode=mode)
        }
        
        if reversed:
            fields = {
                'value': RGBImageField(write_mode=mode),
                'index': IntField()
            }

        writer = DatasetWriter(length, name, fields)

        with writer:
            writer.write_pytorch_dataset(dataset, num_workers=2, chunksize=5)
            
            
        Compiler.set_enabled(False)
        
        loader = Loader(name, batch_size=5, num_workers=2)
        
        import pdb
        
        for res in loader:
            if not reversed:
                index, images  = res
            else:
                images , index = res

            for i, image in zip(index, images):
                if mode == 'raw':
                    assert_that(ch.all((image == (i % 255)).reshape(-1))).is_true()
                else:
                    assert_that(ch.all((image == (i % 255)).reshape(-1))).is_true()
                
def test_simple_raw_image_pipeline():
    create_and_validate(500, 'raw', False)

def test_simple_raw_image_pipeline_rev():
    create_and_validate(500, 'raw', True)

def test_simple_jpg_image_pipeline():
    create_and_validate(500, 'jpg', False)

def test_simple_jpg_image_pipeline_rev():
    create_and_validate(500, 'jpg', True)