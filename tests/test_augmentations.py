import numpy as np
import torch as ch
from torch.utils.data import Dataset
from assertpy import assert_that
from tempfile import NamedTemporaryFile
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset
from ffcv.fields.basics import IntDecoder
from ffcv.fields.rgb_image import SimpleRGBImageDecoder
from ffcv.transforms.cutout import Cutout

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from ffcv.loader import Loader
from ffcv.pipeline.compiler import Compiler
from ffcv.transforms import Squeeze, Cutout, ToTensor, Poison, RandomHorizontalFlip

def run_test(length, pipeline, compile):
    my_dataset = Subset(CIFAR10(root='/tmp', train=True, download=True), range(length))

    with NamedTemporaryFile() as handle:
        name = handle.name
        writer = DatasetWriter(name, {
            'image': RGBImageField(write_mode='smart', 
                                max_resolution=32),
            'label': IntField(),
        }, num_workers=2)

        writer.from_indexed_dataset(my_dataset, chunksize=10)

        Compiler.set_enabled(True)

        loader = Loader(name, batch_size=7, num_workers=2, pipelines={
            'image': pipeline,
            'label': [IntDecoder(), ToTensor(), Squeeze()]
        },
        drop_last=False)
        tot_indices = 0
        tot_images = 0
        for images, label in loader:
            tot_indices += label.shape[0]
            tot_images += images.shape[0]
        assert_that(tot_indices).is_equal_to(len(my_dataset))
        assert_that(tot_images).is_equal_to(len(my_dataset))

def test_flip():
    run_test(100, [
        SimpleRGBImageDecoder(),
        RandomHorizontalFlip(1.0),
        ToTensor()
    ], True)

def test_cutout():
    run_test(100, [
        SimpleRGBImageDecoder(),
        Cutout(8),
        ToTensor()
    ], True)

    run_test(100, [
        SimpleRGBImageDecoder(),
        Cutout(8),
        ToTensor()
    ], False)


def test_poison():
    mask = np.zeros((32, 32, 3))
    # Red sqaure
    mask[:5, :5, 0] = 1
    alpha = np.ones((32, 32))
    run_test(100, [
        SimpleRGBImageDecoder(),
        Poison(mask, alpha, [0, 1, 2]),
        ToTensor()
    ], False)

if __name__ == '__main__':
    test_poison()
