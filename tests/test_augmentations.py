import os
import uuid
import numpy as np
import torch as ch
from torch.utils.data import Dataset
from assertpy import assert_that
from tempfile import NamedTemporaryFile
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image, make_grid
from torch.utils.data import Subset
from ffcv.fields.basics import IntDecoder
from ffcv.fields.rgb_image import SimpleRGBImageDecoder

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from ffcv.loader import Loader
from ffcv.pipeline.compiler import Compiler
from ffcv.transforms import *


SAVE_IMAGES = True
IMAGES_TMP_PATH = '/tmp/ffcv_augtest_output'
if SAVE_IMAGES:
    os.makedirs(IMAGES_TMP_PATH, exist_ok=True)

UNAUGMENTED_PIPELINE=[
    SimpleRGBImageDecoder(),
    ToTensor(),
    ToTorchImage()
]

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

        Compiler.set_enabled(compile)

        loader = Loader(name, batch_size=7, num_workers=2, pipelines={
            'image': pipeline,
            'label': [IntDecoder(), ToTensor(), Squeeze()]
        },
        drop_last=False)

        unaugmented_loader = Loader(name, batch_size=7, num_workers=2, pipelines={
            'image': UNAUGMENTED_PIPELINE,
            'label': [IntDecoder(), ToTensor(), Squeeze()]
        },
        drop_last=False)

        tot_indices = 0
        tot_images = 0
        for (images, labels), (original_images, original_labels) in zip(loader, unaugmented_loader):
            tot_indices += labels.shape[0]
            tot_images += images.shape[0]
            
            for label, original_label in zip(labels, original_labels):
                assert_that(label).is_equal_to(original_label)
            
            if SAVE_IMAGES:
                save_image(make_grid(ch.concat([images, original_images])/255., images.shape[0]), 
                        os.path.join(IMAGES_TMP_PATH, str(uuid.uuid4()) + '.jpeg')
                        )

        assert_that(tot_indices).is_equal_to(len(my_dataset))
        assert_that(tot_images).is_equal_to(len(my_dataset))


def test_cutout():
    for comp in [True, False]:
        run_test(100, [
            SimpleRGBImageDecoder(),
            Cutout(8),
            ToTensor(),
            ToTorchImage()
        ], comp)


def test_flip():
    for comp in [True, False]:
        run_test(100, [
            SimpleRGBImageDecoder(),
            RandomHorizontalFlip(1.0),
            ToTensor(),
            ToTorchImage()
        ], comp)


def test_mixup():
    for comp in [True, False]:
        run_test(100, [
            SimpleRGBImageDecoder(),
            ImageMixup(1, True),
            ToTensor(),
            ToTorchImage()
        ], comp)


def test_poison():
    mask = np.zeros((32, 32, 3))
    # Red sqaure
    mask[:5, :5, 0] = 1
    alpha = np.ones((32, 32))

    for comp in [True, False]:
        run_test(100, [
            SimpleRGBImageDecoder(),
            Poison(mask, alpha, [0, 1, 2]),
            ToTensor(),
            ToTorchImage()
        ], comp)


if __name__ == '__main__':
    # test_cutout()
    test_flip()
    # test_mixup()
    # test_poison()
