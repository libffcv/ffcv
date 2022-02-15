from ffcv.transforms.ops import ToTensor
from ffcv.fields.rgb_image import RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder, CenterCropRGBImageDecoder
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

    def __init__(self, length, size_range):
        self.length = length
        self.size_range = size_range
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index > self.length:
            raise IndexError
        dims = (
            np.random.randint(self.size_range[0], self.size_range[1] + 1),
            np.random.randint(self.size_range[0], self.size_range[1] + 1),
            3
        )
        image_data = ((np.ones(dims) * index) % 255).astype('uint8')
        return index, image_data

def create_and_validate(length, decoder, size, mode='raw', compile=False):

    dataset = DummyDataset(length, (300, 500))

    with NamedTemporaryFile() as handle:
        name = handle.name
        
        fields = {
            'index': IntField(),
            'value': RGBImageField(write_mode=mode)
        }

        writer = DatasetWriter(name, fields, num_workers=2)

        writer.from_indexed_dataset(dataset, chunksize=5)
            
        Compiler.set_enabled(compile)
        
        loader = Loader(name, batch_size=5, num_workers=2,
                        pipelines={
                            'value': [decoder, ToTensor()]
                        })
        
        for index, images in loader:
            for i, image in zip(index, images):
                assert_that(image.shape).is_equal_to((size[0], size[1], 3))
                if mode == 'raw':
                    assert_that(ch.all((image == (i % 255)).reshape(-1))).is_true()
                else:
                    assert_that(ch.all(ch.abs(image - (i % 255)) < 2)).is_true
                

def test_simple_image_decoder_fails_with_variable_images():
    decoder = SimpleRGBImageDecoder()
    assert_that(create_and_validate).raises(TypeError).when_called_with(500, decoder, 32, 'raw')

def test_rrc_decoder_raw():
    size = (160, 160)
    decoder = RandomResizedCropRGBImageDecoder(size)
    create_and_validate(500, decoder, size, 'raw')

def test_rrc_decoder_jpg():
    size = (160, 160)
    decoder = RandomResizedCropRGBImageDecoder(size)
    create_and_validate(500, decoder, size, 'jpg')

def test_rrc_decoder_raw_compiled():
    size = (160, 160)
    decoder = RandomResizedCropRGBImageDecoder(size)
    create_and_validate(500, decoder, size, 'raw', True)

def test_rrc_decoder_jpg_compiled():
    size = (160, 160)
    decoder = RandomResizedCropRGBImageDecoder(size)
    create_and_validate(500, decoder, size, 'jpg', True)

def test_cc_decoder_raw_nc():
    size = (160, 160)
    decoder = CenterCropRGBImageDecoder(size, 224/256)
    create_and_validate(500, decoder, size, 'raw')

def test_cc_decoder_jpg_nc():
    size = (160, 160)
    decoder = CenterCropRGBImageDecoder(size, 224/256)
    create_and_validate(500, decoder, size, 'jpg')

def test_cc_decoder_raw_compiled():
    size = (160, 160)
    decoder = CenterCropRGBImageDecoder(size, 224/256)
    create_and_validate(500, decoder, size, 'raw', True)

def test_cc_decoder_jpg_compiled():
    size = (160, 160)
    decoder = CenterCropRGBImageDecoder(size, 224/256)
    create_and_validate(500, decoder, size, 'jpg', True)


if __name__ == '__main__':
    test_rrc_decoder_jpg()