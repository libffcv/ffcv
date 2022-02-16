import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.operation import Operation, AllocationQuery
from ffcv.transforms import ToTensor, ToTorchImage
from ffcv.writer import DatasetWriter
from dataclasses import replace
from typing import Callable, Optional, Tuple
from ffcv.pipeline.state import State
from ffcv.transforms.utils.fast_crop import rotate, shear, blend, \
    adjust_contrast, posterize
import torchvision.transforms as tv
import cv2
import pytest
import math

class RandAugment(Operation):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        def randaug(im, mem):
            dst, scratch = mem
            for i in my_range(im.shape[0]):
                
                ## TODO actual randaug logic
                
                ## rotate
                deg = np.random.random() * 45.0
                rotate(im[i], dst[i], deg)
                
                ## brighten
                blend(im[i], scratch[i][0], 0.5, dst[i])
                
                ## adjust contrast
                adjust_contrast(im[i], scratch[i][0], 0.5, dst[i])
                
            return dst

        randaug.is_parallel = True
        return randaug

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        return replace(previous_state, shape=(self.size, self.size, 3)), [
            AllocationQuery((self.size, self.size, 3), dtype=np.dtype('uint8')), 
            AllocationQuery((1, self.size, self.size, 3), dtype=np.dtype('uint8'))
        ]

    
@pytest.mark.parametrize('angle', [45])
def test_rotate(angle):
    Xnp = np.random.uniform(0, 255, size=(32, 32, 3)).astype(np.uint8)
    Ynp = np.zeros(Xnp.shape, dtype=np.uint8)
    Xch = torch.tensor(Xnp.astype(np.float32)).permute(2, 0, 1)
    Ych = tv.functional.rotate(Xch, angle).permute(1, 2, 0).numpy().astype(np.uint8)
    rotate(Xnp, Ynp, angle)

    plt.subplot(1, 2, 1)
    plt.imshow(Ynp)
    plt.subplot(1, 2, 2)
    plt.imshow(Ych)
    plt.savefig('example_imgs/rotate-%d.png' % angle)

    assert np.linalg.norm(Ynp.astype(np.float32) - Ych.astype(np.float32)) < 100
    #print(Ynp.min(), Ynp.max(), Ych.min(), Ych.max())


@pytest.mark.parametrize('amt', [0.31])
def test_shear(amt):
    Xnp = np.random.uniform(0, 255, size=(32, 32, 3)).astype(np.uint8)
    Ynp = np.zeros(Xnp.shape, dtype=np.uint8)
    Xch = torch.tensor(Xnp.astype(np.float32)).permute(2, 0, 1)
    Ych = tv.functional.affine(Xch,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[0, math.degrees(math.atan(0.31))],
                interpolation=tv.functional.InterpolationMode.NEAREST,
                fill=0,
                #center=[0, 0],
            ).permute(1, 2, 0).numpy().astype(np.uint8)
    shear(Xnp, Ynp, 0, -0.31)

    plt.subplot(1, 2, 1)
    plt.imshow(Ynp)
    plt.subplot(1, 2, 2)
    plt.imshow(Ych)
    plt.savefig('example_imgs/shear-%f.png' % amt)

    assert np.linalg.norm(Ynp.astype(np.float32) - Ych.astype(np.float32)) < 100
    #print(Ynp.min(), Ynp.max(), Ych.min(), Ych.max())


@pytest.mark.parametrize('amt', [0.5])
def test_brightness(amt):
    Xnp = np.random.uniform(0, 256, size=(32, 32, 3)).astype(np.uint8)
    Ynp = np.zeros(Xnp.shape, dtype=np.uint8)
    Snp = np.zeros(Xnp.shape, dtype=np.uint8)
    Xch = torch.tensor(Xnp.astype(np.float32)/255.).permute(2, 0, 1)
    Ych = (255*tv.functional.adjust_brightness(Xch, amt).permute(1, 2, 0).numpy()).astype(np.uint8)
    blend(Xnp, Snp, amt, Ynp)

    plt.subplot(1, 2, 1)
    plt.imshow(Ynp)
    plt.subplot(1, 2, 2)
    plt.imshow(Ych)
    plt.savefig('example_imgs/brightness-%.2f.png' % amt)

    assert np.linalg.norm(Ynp.astype(np.float32) - Ych.astype(np.float32)) < 100
    #print(Ynp.min(), Ynp.max(), Ych.min(), Ych.max())


@pytest.mark.parametrize('amt', [0.5])
def test_adjust_contrast(amt):
    Xnp = np.random.uniform(0, 256, size=(32, 32, 3)).astype(np.uint8)
    Ynp = np.zeros(Xnp.shape, dtype=np.uint8)
    Snp = np.zeros(Xnp.shape, dtype=np.uint8)
    Xch = torch.tensor(Xnp.astype(np.float32)/255.).permute(2, 0, 1)
    Ych = (255*tv.functional.adjust_contrast(Xch, 0.5).permute(1, 2, 0).numpy()).astype(np.uint8)
    adjust_contrast(Xnp, Snp, 0.5, Ynp)

    plt.subplot(1, 2, 1)
    plt.imshow(Ynp)
    plt.subplot(1, 2, 2)
    plt.imshow(Ych)
    plt.savefig('example_imgs/adjust_contrast-%.2f.png' % amt)

    assert np.linalg.norm(Ynp.astype(np.float32) - Ych.astype(np.float32)) < 100
    #print(Ynp.min(), Ynp.max(), Ych.min(), Ych.max())

@pytest.mark.parametrize('bits', [2])
def test_posterize(bits):
    Xnp = np.random.uniform(0, 256, size=(32, 32, 3)).astype(np.uint8)
    Ynp = np.zeros(Xnp.shape, dtype=np.uint8)
    Xch = torch.tensor(Xnp).permute(2, 0, 1)
    Ych = tv.functional.posterize(Xch, bits).permute(1, 2, 0).numpy()
    posterize(Xnp, bits, Ynp)

    plt.subplot(1, 2, 1)
    plt.imshow(Ynp)
    plt.subplot(1, 2, 2)
    plt.imshow(Ych)
    plt.savefig('example_imgs/posterize-%d.png' % bits)

    print(Ynp.min(), Ynp.max(), Ych.min(), Ych.max())
    assert np.linalg.norm(Ynp.astype(np.float32) - Ych.astype(np.float32)) < 100


if __name__ == '__main__':
    test_rotate(45)
    test_shear(0.31)
    test_brightness(0.5)
    test_adjust_contrast(0.5)
    test_posterize(2)
    BATCH_SIZE = 512
    image_pipelines = {
        'with': [SimpleRGBImageDecoder(), RandAugment(32), ToTensor()],
        'without': [SimpleRGBImageDecoder(), ToTensor()],
        'torchvision': [SimpleRGBImageDecoder(), ToTensor(), ToTorchImage(), tv.RandAugment(num_ops=2, magnitude=10)]
    }

    for name, pipeline in image_pipelines.items():
        loader = Loader('/home/ashert/iclr-followup/ffcv/ffcv/examples/cifar/betons/cifar_train.beton', batch_size=BATCH_SIZE,
                        num_workers=2, order=OrderOption.RANDOM,
                        drop_last=True, pipelines={'image': pipeline})

        for ims, labs in loader: pass
        start_time = time.time()
        for _ in range(5): #(100):
            for ims, labs in loader: pass
        print(f'Method: {name} | Shape: {ims.shape} | Time per epoch: {(time.time() - start_time) / 100:.5f}s')
