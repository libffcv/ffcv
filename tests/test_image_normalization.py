import numpy as np
import torch as ch
from torch.utils.data import Dataset
from assertpy import assert_that
from tempfile import NamedTemporaryFile
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import SimpleRGBImageDecoder
from ffcv.loader import Loader
from ffcv.pipeline.compiler import Compiler
from ffcv.transforms import ToTorchImage, ToTensor, NormalizeImage, View, ToDevice

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
        result = index,image_data
        return result

def test_cpu_normalization():

    dataset = DummyDataset(500, 25, 30)

    with NamedTemporaryFile() as handle:
        name = handle.name
        fields = {
            'index': IntField(),
            'value': RGBImageField(write_mode='raw', jpeg_quality=95)
        }
        writer = DatasetWriter(name, fields, num_workers=2)
        mean = np.array([0, 1, 2])
        std = np.array([1, 10, 20])

        writer.from_indexed_dataset(dataset, chunksize=5)
        loader = Loader(name, batch_size=5, num_workers=2,
        pipelines={
            'value': [
                SimpleRGBImageDecoder(),
                NormalizeImage(mean, std, np.float16),
                View(np.float16),
                ToTensor(),
                ToTorchImage(),
            ]
        })
        ix = 0
        for res in tqdm(loader):
            index, images  = res
            for image in images:
                image = image.numpy()
                ref_image = dataset[ix][1]
                ref_image = ref_image.transpose(2, 0, 1)
                ref_image = ref_image.astype(np.float16)
                ref_image -= mean[:, None, None]
                ref_image /= std[:, None, None]
                assert_that(np.allclose(ref_image, image)).is_true()
                ix += 1

def test_gpu_normalization():

    dataset = DummyDataset(500, 25, 30)

    with NamedTemporaryFile() as handle:
        name = handle.name
        fields = {
            'index': IntField(),
            'value': RGBImageField(write_mode='raw', jpeg_quality=95)
        }
        writer = DatasetWriter(name, fields, num_workers=2)
        mean = np.array([0, 1, 2])
        std = np.array([1, 10, 20])

        writer.from_indexed_dataset(dataset, chunksize=5)

        loader = Loader(name, batch_size=5, num_workers=2,
        pipelines={
            'value': [
                SimpleRGBImageDecoder(),
                ToTensor(),
                ToDevice(ch.device('cuda:0')),
                ToTorchImage(),
                NormalizeImage(mean, std, np.float16),
                View(ch.float16),
            ]
        })
        ix = 0
        for res in tqdm(loader):
            _, images  = res
            for image in images:
                image = image.cpu().numpy()
                ref_image = dataset[ix][1]
                ref_image = ref_image.transpose(2, 0, 1)
                ref_image = ref_image.astype(np.float16)
                ref_image -= mean[:, None, None]
                ref_image /= std[:, None, None]
                assert_that(np.allclose(ref_image, image)).is_true()
                ix += 1
