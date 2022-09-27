'''
Random color operations similar to torchvision.transforms.ColorJitter except not supporting hue
Reference : https://github.com/pytorch/vision/blob/main/torchvision/transforms/functional_tensor.py
'''

import numpy as np

from dataclasses import replace
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.compiler import Compiler



class RandomBrightness(Operation):
    '''
    Randomly adjust image brightness. Operates on raw arrays (not tensors).

    Parameters
    ----------
    magnitude : float
        randomly choose brightness enhancement factor on [max(0, 1-magnitude), 1+magnitude]
    p : float
        probability to apply brightness
    '''
    def __init__(self, magnitude: float, p=0.5):
        super().__init__()
        self.p = p
        self.magnitude = magnitude

    def generate_code(self):
        my_range = Compiler.get_iterator()
        p = self.p
        magnitude = self.magnitude

        def brightness(images, *_):
            def blend(img1, img2, ratio): return (ratio*img1 + (1-ratio)*img2).clip(0, 255).astype(img1.dtype)

            apply_bright = np.random.rand(images.shape[0]) < p
            magnitudes = np.random.uniform(max(0, 1-magnitude), 1+magnitude, images.shape[0])
            for i in my_range(images.shape[0]):
                if apply_bright[i]:
                    images[i] = blend(images[i], 0, magnitudes[i])

            return images

        brightness.is_parallel = True
        return brightness

    def declare_state_and_memory(self, previous_state):
        return (replace(previous_state, jit_mode=True), AllocationQuery(previous_state.shape, previous_state.dtype))



class RandomContrast(Operation):
    '''
    Randomly adjust image contrast. Operates on raw arrays (not tensors).

    Parameters
    ----------
    magnitude : float
        randomly choose contrast enhancement factor on [max(0, 1-magnitude), 1+magnitude]
    p : float
        probability to apply contrast
    '''
    def __init__(self, magnitude, p=0.5):
        super().__init__()
        self.p = p
        self.magnitude = magnitude

    def generate_code(self):
        my_range = Compiler.get_iterator()
        p = self.p
        magnitude = self.magnitude

        def contrast(images, *_):
            def blend(img1, img2, ratio): return (ratio*img1 + (1-ratio)*img2).clip(0, 255).astype(img1.dtype)

            apply_contrast = np.random.rand(images.shape[0]) < p
            magnitudes = np.random.uniform(max(0, 1-magnitude), 1+magnitude, images.shape[0])
            for i in my_range(images.shape[0]):
                if apply_contrast[i]:
                    r, g, b = images[i,:,:,0], images[i,:,:,1], images[i,:,:,2]
                    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).astype(images[i].dtype)
                    images[i] = blend(images[i], l_img.mean(), magnitudes[i])

            return images

        contrast.is_parallel = True
        return contrast

    def declare_state_and_memory(self, previous_state):
        return (replace(previous_state, jit_mode=True), AllocationQuery(previous_state.shape, previous_state.dtype))



class RandomSaturation(Operation):
    '''
    Randomly adjust image color balance. Operates on raw arrays (not tensors).

    Parameters
    ----------
    magnitude : float
        randomly choose color balance enhancement factor on [max(0, 1-magnitude), 1+magnitude]
    p : float
        probability to apply saturation
    '''
    def __init__(self, magnitude, p=0.5):
        super().__init__()
        self.p = p
        self.magnitude = magnitude

    def generate_code(self):
        my_range = Compiler.get_iterator()
        p = self.p
        magnitude = self.magnitude

        def saturation(images, *_):
            def blend(img1, img2, ratio): return (ratio*img1 + (1-ratio)*img2).clip(0, 255).astype(img1.dtype)

            apply_saturation = np.random.rand(images.shape[0]) < p
            magnitudes = np.random.uniform(max(0, 1-magnitude), 1+magnitude, images.shape[0])
            for i in my_range(images.shape[0]):
                if apply_saturation[i]:
                    r, g, b = images[i,:,:,0], images[i,:,:,1], images[i,:,:,2]
                    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).astype(images[i].dtype)
                    l_img3 = np.zeros_like(images[i])
                    for j in my_range(images[i].shape[-1]):
                        l_img3[:,:,j] = l_img
                    images[i] = blend(images[i], l_img3, magnitudes[i])

            return images

        saturation.is_parallel = True
        return saturation

    def declare_state_and_memory(self, previous_state):
        return (replace(previous_state, jit_mode=True), AllocationQuery(previous_state.shape, previous_state.dtype))
