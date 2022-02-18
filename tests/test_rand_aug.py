import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToTorchImage, RandAugment
from ffcv.writer import DatasetWriter
from dataclasses import replace
from typing import Callable, Optional, Tuple
from ffcv.pipeline.state import State
from ffcv.transforms.utils.fast_crop import rotate, shear, blend, \
    adjust_contrast, posterize, invert, solarize, equalize, fast_equalize, \
    autocontrast, sharpen, adjust_saturation, translate, adjust_brightness
import torchvision.transforms as tv
import cv2
import pytest
import math
    
@pytest.mark.parametrize('angle', [45, -30])
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


@pytest.mark.parametrize('amt', [0.31, -0.31])
def test_shear(amt):
    Xnp = np.random.uniform(0, 255, size=(32, 32, 3)).astype(np.uint8)
    Ynp = np.zeros(Xnp.shape, dtype=np.uint8)
    Xch = torch.tensor(Xnp.astype(np.float32)).permute(2, 0, 1)
    Ych = tv.functional.affine(Xch,
                angle=0.0,
                translate=[0, 0],
                scale=1.0,
                shear=[0, math.degrees(math.atan(amt))],
                interpolation=tv.functional.InterpolationMode.NEAREST,
                fill=0,
                #center=[0, 0],
            ).permute(1, 2, 0).numpy().astype(np.uint8)
    shear(Xnp, Ynp, 0, -amt)

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
    adjust_brightness(Xnp, Snp, amt, Ynp)

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
    Ych = (255*tv.functional.adjust_contrast(Xch, amt).permute(1, 2, 0).numpy()).astype(np.uint8)
    adjust_contrast(Xnp, Snp, amt, Ynp)

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

    assert np.linalg.norm(Ynp.astype(np.float32) - Ych.astype(np.float32)) < 100


def test_invert():
    Xnp = np.random.uniform(0, 256, size=(32, 32, 3)).astype(np.uint8)
    Xnp[5:9,5:9,:] = 0
    Ynp = np.zeros(Xnp.shape, dtype=np.uint8)
    Xch = torch.tensor(Xnp).permute(2, 0, 1)
    Ych = tv.functional.invert(Xch).permute(1, 2, 0).numpy()
    invert(Xnp, Ynp)

    plt.subplot(1, 2, 1)
    plt.imshow(Ynp)
    plt.subplot(1, 2, 2)
    plt.imshow(Ych)
    plt.savefig('example_imgs/invert.png')

    assert np.linalg.norm(Ynp.astype(np.float32) - Ych.astype(np.float32)) < 100


@pytest.mark.parametrize('threshold', [9])
def test_solarize(threshold):
    Xnp = np.random.uniform(0, 256, size=(32, 32, 3)).astype(np.uint8)
    Xnp[5:9,5:9,:] = 0
    Xnp[10:15,10:15,:] = 8
    Xnp[27:31,27:31,:] = 9
    Ynp = np.zeros(Xnp.shape, dtype=np.uint8)
    Xch = torch.tensor(Xnp).permute(2, 0, 1)
    Ych = tv.functional.solarize(Xch, threshold).permute(1, 2, 0).numpy()
    solarize(Xnp, threshold, Ynp)

    plt.subplot(1, 2, 1)
    plt.imshow(Ynp)
    plt.subplot(1, 2, 2)
    plt.imshow(Ych)
    plt.savefig('example_imgs/solarize-%d.png' % threshold)

    assert np.linalg.norm(Ynp.astype(np.float32) - Ych.astype(np.float32)) < 100


def test_equalize():
    Xnp = np.random.uniform(0, 256, size=(32, 32, 3)).astype(np.uint8)
    #Xnp = cv2.imread('example_imgs/0249.png')
    Xnp[5:9,5:9,:] = 0
    Ynp = np.zeros(Xnp.shape, dtype=np.uint8)
    #Snp_chw = np.zeros((3, 32, 32), dtype=np.uint8)
    Snp = np.zeros((3, 256), dtype=np.int16)
    Xch = torch.tensor(Xnp).permute(2, 0, 1)
    Ych = tv.functional.equalize(Xch).permute(1, 2, 0).numpy()
    #fast_equalize(Xnp, Snp_chw, Ynp)
    equalize(Xnp, Snp, Ynp)

    plt.subplot(2, 2, 1)
    plt.imshow(Xnp)
    plt.subplot(2, 2, 2)
    plt.imshow(Ynp)
    plt.subplot(2, 2, 3)
    plt.imshow(Xch.permute(1, 2, 0).numpy())
    plt.subplot(2, 2, 4)
    plt.imshow(Ych)
    plt.savefig('example_imgs/equalize.png')
    
    assert np.linalg.norm(Ynp.astype(np.float32) - Ych.astype(np.float32)) < 100

    
def test_autocontrast():
    Xnp = np.random.uniform(0, 256, size=(32, 32, 3)).astype(np.uint8)
    #Xnp = cv2.imread('example_imgs/0249.png')
    Xnp[5:9,5:9,:] = 0
    Ynp = np.zeros(Xnp.shape, dtype=np.uint8)
    Snp = np.zeros((32, 32, 3), dtype=np.float32)
    Xch = torch.tensor(Xnp).permute(2, 0, 1)
    Ych = tv.functional.autocontrast(Xch).permute(1, 2, 0).numpy()
    autocontrast(Xnp, Snp, Ynp)

    plt.subplot(2, 2, 1)
    plt.imshow(Xnp)
    plt.subplot(2, 2, 2)
    plt.imshow(Ynp)
    plt.subplot(2, 2, 3)
    plt.imshow(Xch.permute(1, 2, 0).numpy())
    plt.subplot(2, 2, 4)
    plt.imshow(Ych)
    plt.savefig('example_imgs/autocontrast.png')
    
    assert np.linalg.norm(Ynp.astype(np.float32) - Ych.astype(np.float32)) < 100


@pytest.mark.parametrize('amt', [2.0])
def test_sharpen(amt):
    Xnp = np.random.uniform(0, 256, size=(32, 32, 3)).astype(np.uint8)
    #Xnp = cv2.imread('example_imgs/0249.png')
    Ynp = np.zeros(Xnp.shape, dtype=np.uint8)
    Snp = np.zeros(Xnp.shape, dtype=np.uint8)
    Xch = torch.tensor(Xnp).permute(2, 0, 1)
    Ych = tv.functional.adjust_sharpness(Xch, amt).permute(1, 2, 0).numpy()
    sharpen(Xnp, Ynp, amt)

    plt.subplot(1, 2, 1)
    plt.imshow(Ynp)
    plt.subplot(1, 2, 2)
    plt.imshow(Ych)
    plt.savefig('example_imgs/sharpen-%.2f.png' % amt)

    assert np.linalg.norm(Ynp.astype(np.float32) - Ych.astype(np.float32)) < 100


@pytest.mark.parametrize('amt', [0.5, 1.5])
def test_adjust_saturation(amt):
    Xnp = np.random.uniform(0, 256, size=(32, 32, 3)).astype(np.uint8)
    #Xnp = cv2.imread('example_imgs/0249.png')
    Ynp = np.zeros(Xnp.shape, dtype=np.uint8)
    Snp = np.zeros(Xnp.shape, dtype=np.uint8)
    Xch = torch.tensor(Xnp.astype(np.float32)/255.).permute(2, 0, 1)
    Ych = (255*tv.functional.adjust_saturation(Xch, amt).permute(1, 2, 0).numpy()).astype(np.uint8)
    adjust_saturation(Xnp, Snp, amt, Ynp)

    plt.subplot(2, 2, 1)
    plt.imshow(Xnp)
    plt.subplot(2, 2, 2)
    plt.imshow(Ynp)
    plt.subplot(2, 2, 3)
    plt.imshow(Xch.permute(1, 2, 0).numpy())
    plt.subplot(2, 2, 4)
    plt.imshow(Ych)
    plt.savefig('example_imgs/adjust_saturation-%.2f.png' % amt)

    assert np.linalg.norm(Ynp.astype(np.float32) - Ych.astype(np.float32)) < 100
    #print(Ynp.min(), Ynp.max(), Ych.min(), Ych.max())


@pytest.mark.parametrize('amt', [(4, 0), (0, 4), (-4, 0), (0, -4)])
def test_translate(amt):
    Xnp = np.random.uniform(0, 255, size=(32, 32, 3)).astype(np.uint8)
    Ynp = np.zeros(Xnp.shape, dtype=np.uint8)
    Xch = torch.tensor(Xnp.astype(np.float32)).permute(2, 0, 1)
    Ych = tv.functional.affine(Xch,
                angle=0.0,
                translate=[amt[0], amt[1]],
                scale=1.0,
                shear=[0, 0],
                interpolation=tv.functional.InterpolationMode.NEAREST,
                fill=0,
                #center=[0, 0],
            ).permute(1, 2, 0).numpy().astype(np.uint8)
    translate(Xnp, Ynp, amt[0], amt[1])

    plt.subplot(1, 2, 1)
    plt.imshow(Ynp)
    plt.subplot(1, 2, 2)
    plt.imshow(Ych)
    plt.savefig('example_imgs/translate-%d-%d.png' % amt)

    assert np.linalg.norm(Ynp.astype(np.float32) - Ych.astype(np.float32)) < 100
    

if __name__ == '__main__':
#     test_rotate(45)
#     test_shear(0.31)
#     test_brightness(0.5)
#     test_adjust_contrast(0.5)
#     test_posterize(2)
#     test_invert()
#     test_solarize(9)
#     test_equalize()
#     test_autocontrast()
#     test_sharpen(2.0)
#     test_adjust_saturation(0.5)
#     test_translate((4, 0))
    
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

        import matplotlib.pyplot as plt
        idx = 0
        for ims, labs in loader: pass
#             print('a')
#             if name == 'with':
#                 for i in range(5):
#                     for j in range(5):
#                         plt.subplot(5, 5, i * 5 + j + 1)
#                         plt.imshow(ims[i * 5 + j].numpy())
#                 plt.savefig('scratch/%d.png'%idx)
#                 idx+=1
        start_time = time.time()
        for _ in range(5): #(100):
            for ims, labs in loader: pass
        print(f'Method: {name} | Shape: {ims.shape} | Time per epoch: {(time.time() - start_time) / 100:.5f}s')
