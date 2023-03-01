import os
import uuid
import numpy as np
import torch as ch
from torch.utils.data import Dataset
from torchvision import transforms as tvt
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

def run_test(length, pipeline, should_compile=False, aug_name=''):
    my_dataset = Subset(CIFAR10(root='/tmp', train=True, download=True), range(length))

    with NamedTemporaryFile() as handle:
        name = handle.name
        writer = DatasetWriter(name, {
            'image': RGBImageField(write_mode='smart', 
                                max_resolution=32),
            'label': IntField(),
        }, num_workers=2)

        writer.from_indexed_dataset(my_dataset, chunksize=10)

        Compiler.set_enabled(should_compile)

        loader = Loader(name, batch_size=7, num_workers=2, pipelines={
            'image': pipeline,
            'label': [IntDecoder(), ToTensor(), Squeeze()]
        },
        drop_last=False)

        unaugmented_loader = Loader(name, batch_size=7, num_workers=2, pipelines={
            'image': UNAUGMENTED_PIPELINE,
            'label': [IntDecoder(), ToTensor(), Squeeze()]
        }, drop_last=False)

        tot_indices = 0
        tot_images = 0
        for it_num, ((images, labels), (original_images, original_labels)) in enumerate(zip(loader, unaugmented_loader)):
            tot_indices += labels.shape[0]
            tot_images += images.shape[0]
            
            for label, original_label in zip(labels, original_labels):
                assert_that(label).is_equal_to(original_label)
            
            if SAVE_IMAGES and it_num == 0:
                save_image(make_grid(ch.concat([images, original_images])/255., images.shape[0]), 
                        os.path.join(IMAGES_TMP_PATH, aug_name + '-' + str(uuid.uuid4()) + '.jpeg'))

        assert_that(tot_indices).is_equal_to(len(my_dataset))
        assert_that(tot_images).is_equal_to(len(my_dataset))

def test_cutout():
    for comp in [True, False]:
        run_test(100, [
            SimpleRGBImageDecoder(),
            Cutout(8),
            ToTensor(),
            ToTorchImage()
        ], comp, 'cutout')


def test_flip():
    for comp in [True, False]:
        run_test(100, [
            SimpleRGBImageDecoder(),
            RandomHorizontalFlip(1.0),
            ToTensor(),
            ToTorchImage()
        ], comp, 'flip')


def test_module_wrapper():
    for comp in [True, False]:
        run_test(100, [
            SimpleRGBImageDecoder(),
            ToTensor(),
            ToTorchImage(),
            ModuleWrapper(tvt.Grayscale(3)),
        ], comp, 'module')


def test_mixup():
    for comp in [True, False]:
        run_test(100, [
            SimpleRGBImageDecoder(),
            ImageMixup(.5, False),
            ToTensor(),
            ToTorchImage()
        ], comp, 'mixup')


def test_poison():
    mask = np.zeros((32, 32, 3))
    # Red sqaure
    mask[:5, :5, 0] = 1
    alpha = np.ones((32, 32))

    for comp in [True, False]:
        run_test(100, [
            SimpleRGBImageDecoder(),
            Poison(mask, alpha, list(range(100))),
            ToTensor(),
            ToTorchImage()
        ], comp, 'poison')

def test_random_resized_crop():
    for comp in [True, False]:
        run_test(100, [
            SimpleRGBImageDecoder(),
            RandomResizedCrop(scale=(0.08, 1.0), 
                            ratio=(0.75, 4/3),
                            size=32),
            ToTensor(),
            ToTorchImage()
        ], comp, 'rrc')


def test_translate():
    for comp in [True, False]:
        run_test(100, [
            SimpleRGBImageDecoder(),
            RandomTranslate(padding=10),
            ToTensor(),
            ToTorchImage()
        ], comp, 'translate')


## Torchvision Transforms
def test_torchvision_greyscale():
    run_test(100, [
        SimpleRGBImageDecoder(),
        ToTensor(),
        ToTorchImage(),
        tvt.Grayscale(3),
        ], aug_name='tv_grey')

def test_torchvision_centercrop_pad():
    run_test(100, [
        SimpleRGBImageDecoder(),
        ToTensor(),
        ToTorchImage(),
        tvt.CenterCrop(10),
        tvt.Pad(11)
        ], aug_name='tv_crop_pad')

def test_torchvision_random_affine():
    run_test(100, [
        SimpleRGBImageDecoder(),
        ToTensor(),
        ToTorchImage(),
        tvt.RandomAffine(25),
        ], aug_name='tv_random_affine')

def test_torchvision_random_crop():
    run_test(100, [
        SimpleRGBImageDecoder(),
        ToTensor(),
        ToTorchImage(),
        tvt.Pad(10),
        tvt.RandomCrop(size=32),
        ], aug_name='tv_randcrop')

def test_torchvision_color_jitter():
    run_test(100, [
        SimpleRGBImageDecoder(),
        ToTensor(),
        ToTorchImage(),
        tvt.ColorJitter(.5, .5, .5, .5)
    ], aug_name='tv_colorjitter')
