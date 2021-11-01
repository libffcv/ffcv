"""
Instructions:
- Clone the main branch of the repo
- python setup.py install there
- Come back here and run this with pytest -k
"""


import numpy as np
from tqdm import tqdm
from time import time
from torchvision import transforms
import torch as ch

from fastercv.loader import FastImageNetDataset

from ffcv.pipeline.compiler import Compiler
from ffcv.transforms.ops import ToTorchImage
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import Cutout, RandomHorizontalFlip, ToTensor, Collate, ToDevice, Squeeze, Convert
from ffcv.fields.rgb_image import RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

batch_size = 1024
n = 3

def measure(ds):
    start = time()
    for _ in ds:
        pass
    return time() - start

def test_old_loader():
    data_source = '/tmp/fastercv-imagenet-train-256.dat'
    cropper = transforms.RandomResizedCrop(224)

    ds = FastImageNetDataset(data_source,
                               batch_size,
                               cropper,
                               res_scheduler=lambda x: 224,
                               num_workers=16,
                               pin_memory=True,
                               drop_last=True)

    # Warm up
    measure(ds)
    print(np.median([measure(ds) for _ in range(n)]))
    return True

def test_new_loader():
    loader = Loader('/dev/shm/imagenet_train.beton',
                batch_size=1024,
                num_workers=96,
                order=OrderOption.SEQUENTIAL,
                pipelines={
                    'image': [
                        RandomResizedCropRGBImageDecoder((224, 224)), 
                        ToTensor(),
                        ToDevice(ch.device('cuda:0'), non_blocking=True),
                        ToTorchImage(),
                        Convert(ch.float16),
                    ],
                    'label': [
                        IntDecoder(),
                        ToTensor(),
                        ToDevice(ch.device('cuda:0')),
                        Squeeze()
                        ]
                })
    
    for _ in tqdm(loader):
        pass
    # measure(loader)
    # print(np.median([measure(loader) for _ in range(n)]))
    return True

if __name__ == '__main__':
    test_new_loader()
