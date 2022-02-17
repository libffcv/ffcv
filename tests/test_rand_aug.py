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
    adjust_contrast, posterize, invert, solarize, equalize, fast_equalize, \
    autocontrast, sharpen, adjust_saturation, translate, adjust_brightness
import torchvision.transforms as tv
import cv2
import pytest
import math

class RandAugment(Operation):
    def __init__(self, 
                 size: int = 32, 
                 num_ops: int = 2, 
                 magnitude: int = 9, 
                 num_magnitude_bins: int = 31):
        super().__init__()
        self.size = size
        self.num_ops = num_ops
        self.magnitude = magnitude
        num_bins = num_magnitude_bins
        # index, name (for readability); bins, sign multiplier
        # those with a -1 can have negative magnitude with probability 0.5
        self.op_table = [
            (0, "Identity", np.array(0.0), 1),
            (1, "ShearX", np.linspace(0.0, 0.3, num_bins), -1),
            (2, "ShearY", np.linspace(0.0, 0.3, num_bins), -1),
            (3, "TranslateX", np.linspace(0.0, 150.0 / 331.0 * size, num_bins), -1),
            (4, "TranslateY", np.linspace(0.0, 150.0 / 331.0 * size, num_bins), -1),
            (5, "Rotate", np.linspace(0.0, 30.0, num_bins), -1),
            (6, "Brightness", np.linspace(0.0, 0.9, num_bins), -1),
            (7, "Color", np.linspace(0.0, 0.9, num_bins), -1),
            (8, "Contrast", np.linspace(0.0, 0.9, num_bins), -1),
            (9, "Sharpness", np.linspace(0.0, 0.9, num_bins), -1),
            (10, "Posterize", 8 - (np.arange(num_bins) / ((num_bins - 1) / 4)).round().astype('uint8'), 1),
            (11, "Solarize", np.linspace(255.0, 0.0, num_bins), 1),
            (12, "AutoContrast", np.array(0.0), 1),
            (13, "Equalize", np.array(0.0), 1),
        ]

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        op_table = self.op_table
        magnitudes = np.array([(op[2][self.magnitude] if op[2].ndim > 0 else 0) for op in self.op_table])
        is_signed = np.array([op[3] for op in self.op_table])
        num_ops = self.num_ops
#         for i in range(len(magnitudes)):
#             print(i, op_table[i][1], '%.3f'%magnitudes[i])
        def randaug(im, mem):
            dst, scratch, lut, scratchf = mem
            for i in my_range(im.shape[0]):
                for n in range(num_ops):
                    if n == 0:
                        src = im
                    else:
                        src = dst
                        
                    idx = np.random.randint(low=0, high=13+1)
                    mag = magnitudes[idx]
                    if np.random.random() < 0.5:
                        mag = mag * is_signed[idx] 

                    # Not worth fighting numba at the moment.
                    # TODO
                    if idx == 0:
                        dst[i][:] = src[i]
                    
                    if idx == 1: # ShearX (0.004)
                        shear(src[i], dst[i], mag, 0)

                    if idx == 2: # ShearY
                        shear(src[i], dst[i], 0, mag)

                    if idx == 3: # TranslateX
                        translate(src[i], dst[i], int(mag), 0)

                    if idx == 4: # TranslateY
                        translate(src[i], dst[i], 0, int(mag))

                    if idx == 5: # Rotate
                        rotate(src[i], dst[i], mag)

                    if idx == 6: # Brightness
                        adjust_brightness(src[i], scratch[i][0], 1.0 + mag, dst[i])

                    if idx == 7: # Color
                        adjust_saturation(src[i], scratch[i][0], 1.0 + mag, dst[i])

                    if idx == 8: # Contrast
                        adjust_contrast(src[i], scratch[i][0], 1.0 + mag, dst[i])

                    if idx == 9: # Sharpness
                        sharpen(src[i], dst[i], 1.0 + mag)

                    if idx == 10: # Posterize
                        posterize(src[i], int(mag), dst[i])

                    if idx == 11: # Solarize
                        solarize(src[i], mag, dst[i])

                    if idx == 12: # AutoContrast (TODO: takes 0.04s -> 0.052s) (+0.01s)
                        autocontrast(src[i], scratchf[i][0], dst[i])
                    
                    if idx == 13: # Equalize (TODO: +0.008s)
                        equalize(src[i], lut[i], dst[i])
                
            return dst

        randaug.is_parallel = True
        return randaug

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        return replace(previous_state, shape=(self.size, self.size, 3)), [
            AllocationQuery((self.size, self.size, 3), dtype=np.dtype('uint8')), 
            AllocationQuery((1, self.size, self.size, 3), dtype=np.dtype('uint8')),
            AllocationQuery((3, 256), dtype=np.dtype('int16')),
            AllocationQuery((1, self.size, self.size, 3), dtype=np.dtype('float32')),
        ]

    
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
