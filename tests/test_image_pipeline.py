import numpy as np
import torch as ch
from torch.utils.data import Dataset
from assertpy import assert_that
from tempfile import NamedTemporaryFile
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset

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
            'value': RGBImageField(write_mode=mode, max_resolution=32)
        })

        with writer:
            writer.write_pytorch_dataset(dataset, num_workers=2, chunksize=5)
            
        Compiler.set_enabled(False)
        
        loader = Loader(name, batch_size=5, num_workers=2)
        
        for index, images in loader:
            for i, image in zip(index, images):
                assert_that(ch.all((image == (i % 255)).reshape(-1))).is_true()
                
def make_and_read_cifar_subset(length):
    my_dataset = Subset(CIFAR10(root='/tmp', train=True, download=True), range(length))

    with NamedTemporaryFile() as handle:
        name = handle.name
        writer = DatasetWriter(len(my_dataset), name, {
            'image': RGBImageField(write_mode='smart', 
                                max_resolution=32),
            'label': IntField(),
        })

        with writer:
            writer.write_pytorch_dataset(my_dataset, num_workers=2, chunksize=10)

        Compiler.set_enabled(False)
        
        loader = Loader(name, batch_size=5, num_workers=2)
        
        for index, images in loader:
            pass

def test_simple_raw_image_pipeline():
    create_and_validate(500, 'raw')

def test_cifar_subset():
    make_and_read_cifar_subset(200)
