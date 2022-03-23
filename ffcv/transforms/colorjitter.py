"""
ColorJitter
Code for Brightness, Contrast and Saturation adapted from
https://github.com/pytorch/vision/blob/main/torchvision/transforms/functional_tensor.py
Code for Hue adapted from:
https://sanje2v.wordpress.com/2021/01/11/accelerating-data-transforms/
https://stackoverflow.com/questions/8507885
"""
import numpy as np
from numpy.random import rand
from typing import Callable, Optional, Tuple
from dataclasses import replace
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.compiler import Compiler
import numbers
import numba as nb
from math import sqrt,cos,sin,radians

class ColorJitter(Operation):
    """Add ColorJitter with probability jitter_prob.
    Operates on raw arrays (not tensors).

    Parameters
    ----------
    jitter_prob : float, The probability with which to apply ColorJitter.
    brightness (float or tuple of float (min, max)): How much to jitter brightness.
        brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
        or the given [min, max]. Should be non negative numbers.
    contrast (float or tuple of float (min, max)): How much to jitter contrast.
        contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
        or the given [min, max]. Should be non negative numbers.
    saturation (float or tuple of float (min, max)): How much to jitter saturation.
        saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
        or the given [min, max]. Should be non negative numbers.
    hue (float or tuple of float (min, max)): How much to jitter hue.
        hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
        Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, jitter_prob, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
        self.jitter_prob = jitter_prob
        assert self.jitter_prob >= 0 and self.jitter_prob <= 1

    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        jitter_prob = self.jitter_prob
        apply_brightness = (self.brightness is not None)
        if apply_brightness:
            brightness_min, brightness_max = self.brightness
        apply_contrast = (self.contrast is not None)
        if apply_contrast:
            contrast_min, contrast_max = self.contrast
        apply_saturation = (self.saturation is not None)
        if apply_saturation:
            saturation_min, saturation_max = self.saturation
        apply_hue = (self.hue is not None)
        if apply_hue:
            hue_min, hue_max = self.hue
            
        def color_jitter(images, dst):
            should_jitter = rand(images.shape[0]) < jitter_prob
            for i in my_range(images.shape[0]):
                if should_jitter[i]:
                    img = images[i]
                    # Brightness
                    if apply_brightness:
                        ratio_brightness = np.random.uniform(brightness_min, brightness_max)
                        img = ratio_brightness * img + (1.0 - ratio_brightness) * img * 0
                        img = np.clip(img, 0, 255)
                    
                    # Contrast
                    if apply_contrast:
                        ratio_contrast = np.random.uniform(contrast_min, contrast_max)
                        gray = 0.2989 * img[:,:,0:1] + 0.5870 * img[:,:,1:2] + 0.1140 * img[:,:,2:3]
                        img = ratio_contrast * img + (1.0 - ratio_contrast) * gray.mean()
                        img = np.clip(img, 0, 255)
                    
                    # Saturation
                    if apply_saturation:
                        ratio_saturation = np.random.uniform(saturation_min, saturation_max)
                        dst[i] = 0.2989 * img[:,:,0:1] + 0.5870 * img[:,:,1:2] + 0.1140 * img[:,:,2:3]
                        img = ratio_saturation * img + (1.0 - ratio_saturation) * dst[i]
                        img = np.clip(img, 0, 255)

                    # Hue
                    if apply_hue:
                        img = img / 255.0
                        hue_factor = np.random.uniform(hue_min, hue_max)
                        hue_factor_radians = hue_factor * 2.0 * np.pi
                        cosA = np.cos(hue_factor_radians)
                        sinA = np.sin(hue_factor_radians)
                        hue_rotation_matrix =\
                        [[cosA + (1.0 - cosA) / 3.0, 1./3. * (1.0 - cosA) - np.sqrt(1./3.) * sinA, 1./3. * (1.0 - cosA) + np.sqrt(1./3.) * sinA],
                        [1./3. * (1.0 - cosA) + np.sqrt(1./3.) * sinA, cosA + 1./3.*(1.0 - cosA), 1./3. * (1.0 - cosA) - np.sqrt(1./3.) * sinA],
                        [1./3. * (1.0 - cosA) - np.sqrt(1./3.) * sinA, 1./3. * (1.0 - cosA) + np.sqrt(1./3.) * sinA, cosA + 1./3. * (1.0 - cosA)]]
                        hue_rotation_matrix = np.array(hue_rotation_matrix, dtype=img.dtype)
                        for row in nb.prange(img.shape[0]):
                            for col in nb.prange(img.shape[1]):
                                r, g, b = img[row, col, :]
                                img[row, col, 0] = r * hue_rotation_matrix[0, 0] + g * hue_rotation_matrix[0, 1] + b * hue_rotation_matrix[0, 2]
                                img[row, col, 1] = r * hue_rotation_matrix[1, 0] + g * hue_rotation_matrix[1, 1] + b * hue_rotation_matrix[1, 2]
                                img[row, col, 2] = r * hue_rotation_matrix[2, 0] + g * hue_rotation_matrix[2, 1] + b * hue_rotation_matrix[2, 2]
                        img = np.asarray(np.clip(img * 255., 0, 255), dtype=np.uint8)
                    dst[i] = img
                else:
                    dst[i] = images[i]
            return dst

        color_jitter.is_parallel = True
        return color_jitter

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), AllocationQuery(shape=previous_state.shape, dtype=previous_state.dtype))

