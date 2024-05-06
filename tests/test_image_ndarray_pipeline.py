import numpy as np
import torch as ch
from torch.utils.data import Dataset
from assertpy import assert_that
from tempfile import NamedTemporaryFile
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset
import torch
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField, NDArrayField, FloatField
from ffcv.loader import Loader
from ffcv.pipeline.compiler import Compiler

from ffcv.fields.ndarray import NDArrayDecoder
from ffcv.fields.decoders import SimpleRGBImageDecoder
from ffcv.transforms import ToTensor, ToDevice


class DummyDataset(Dataset):

    def __init__(self, length, label_dtype, height, width):
        self.length = length
        self.height = height
        self.width = width
        self.label_dtype = label_dtype
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index > self.length:
            raise IndexError
        dims = (self.height, self.width, 3)
        image_data = ((np.ones(dims) * index) % 255).astype('uint8')
        if self.label_dtype == np.ndarray:
            label = np.ones(2, dtype=np.float32) * index
        else:
            label = index
        result = image_data, label
        return result

def create_and_validate_ndarray(length, dtype, mode='raw'):

    dataset = DummyDataset(length=length, label_dtype=np.ndarray, height=500, width=300)

    with NamedTemporaryFile() as handle:
        name = handle.name
        fields = {
            'value': RGBImageField(write_mode=mode, jpeg_quality=95),
            'label': NDArrayField(shape=(2,), dtype=np.dtype('float32')),
        }
        writer = DatasetWriter(name, fields, num_workers=4)
        writer.from_indexed_dataset(dataset, chunksize=5)
        Compiler.set_enabled(False)
        loader = Loader(name, batch_size=5, num_workers=2)
        labels = []
        for images, label in loader:
            labels.append(label[:, 0])
        expected = np.arange(length).astype(np.float32)
        labels = torch.concat(labels).numpy()
        np.testing.assert_array_equal(expected, labels)

def create_and_validate_int(length, dtype, mode='raw'):

    dataset = DummyDataset(length=length, label_dtype=int, height=500, width=300)

    with NamedTemporaryFile() as handle:
        name = handle.name
        fields = {
            'value': RGBImageField(write_mode=mode, jpeg_quality=95),
            'label': IntField(),
        }
        writer = DatasetWriter(name, fields, num_workers=4)
        writer.from_indexed_dataset(dataset, chunksize=5)
        Compiler.set_enabled(False)
        loader = Loader(name, batch_size=5, num_workers=2)
        labels = []
        for images, label in loader:
            labels.append(label[:, 0])
        expected = np.arange(length).astype(np.float32)
        labels = torch.concat(labels).numpy().astype(np.float32)
        np.testing.assert_array_equal(expected, labels)

def test_simple_jpg_image_pipeline_ndarray():
    create_and_validate_ndarray(100, 'jpg')

def test_simple_jpg_image_pipeline_int():
    create_and_validate_int(100, 'jpg')

