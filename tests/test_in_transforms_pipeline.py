import numpy as np
from tempfile import NamedTemporaryFile
from numpy.lib.npyio import load

import torch as ch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from ffcv.loader import Loader, OrderOption
from ffcv.transforms.ops import ToTorchImage
from ffcv.transforms import RandomHorizontalFlip, ToTensor, Convert
from ffcv.transforms import Squeeze, ToDevice
from ffcv.fields.rgb_image import RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from ffcv.loader import Loader
from ffcv.pipeline.compiler import Compiler

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
DEFAULT_CROP_RATIO = 224/256

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

def create_train_loader(name, batch_size, num_workers):
    loader = Loader(name,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=OrderOption.QUASI_RANDOM,
                    pipelines={
                        "image": [
                            RandomResizedCropRGBImageDecoder((224, 224)),
                            RandomHorizontalFlip(0.5),
                            ToTensor(),
                            ToDevice(ch.device('cuda:0')),
                            ToTorchImage(),
                            Convert(ch.float16),
                            Normalize((IMAGENET_MEAN * 255).tolist(), (IMAGENET_STD * 255).tolist()),
                        ],
                        "label": [IntDecoder(), ToTensor(), Squeeze(), ToDevice(ch.device("cuda:0"))]
                    }
            )
    return loader

def create_val_loader(name, batch_size, num_workers):
    loader = Loader(name,
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.QUASI_RANDOM,
            pipelines={
                "image": [
                    CenterCropRGBImageDecoder(output_size=(224,224), ratio=DEFAULT_CROP_RATIO),
                    ToTensor(),
                    ToDevice(ch.device('cuda:0')),
                    ToTorchImage(),
                    Convert(ch.float16),
                    Normalize((IMAGENET_MEAN * 255).tolist(), (IMAGENET_STD * 255).tolist()),
                ],
                "label": [IntDecoder(), ToTensor(), Squeeze(), ToDevice(ch.device("cuda:0"), non_blocking=True)]
            }
    )
    return loader

def loop_loader(length=10, batch_size=2, num_workers=8, reversed=False, train=True):

    dataset = DummyDataset(length, 500, 300, reversed=reversed)

    with NamedTemporaryFile() as handle:
        name = handle.name

        fields = {
            'label': IntField(),
            'image': RGBImageField(write_mode="raw", max_resolution=256, smart_threshold=3)
        }

        if reversed:

            fields = {
                'image': RGBImageField(write_mode="raw", max_resolution=256, smart_threshold=3),
                'label': IntField()
            }


        writer = DatasetWriter(length, name, fields)

        with writer:
            writer.write_pytorch_dataset(dataset, num_workers=2, chunksize=5)

        Compiler.set_enabled(True)

        if train:
            loader = create_train_loader(name, batch_size, num_workers)
        else:
            loader = create_val_loader(name, batch_size, num_workers)

        for X, y in loader:
            pass

def test_train_loader():
    loop_loader(length=100, batch_size=16, train=True)

def test_val_loader():
    loop_loader(length=100, batch_size=16, train=False)