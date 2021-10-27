import numpy as np
import torch as ch
from torch.utils.data import Dataset
from assertpy import assert_that
from tempfile import NamedTemporaryFile
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset
from ffcv.fields.basics import IntDecoder
from ffcv.fields.rgb_image import RGBImageDecoder
from ffcv.transforms.cutout import Cutout

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from ffcv.loader import Loader
from ffcv.pipeline.compiler import Compiler
from ffcv.transforms import Squeeze, Cutout, ToTensor, ToDevice

def run_test(length, pipeline, compile):
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

        Compiler.set_enabled(True)
    
        loader = Loader(name, batch_size=5, num_workers=2, pipelines={
            'image': pipeline,
            'label': [IntDecoder(), ToTensor(), Squeeze()]
        })
        for index, images in loader:
            pass

def test_cutout():
    run_test(100, [
        RGBImageDecoder(),
        Cutout(8),
        ToTensor()
    ], True)

    run_test(100, [
        RGBImageDecoder(),
        Cutout(8),
        ToTensor()
    ], False)