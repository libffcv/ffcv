"""
Random resized crop, similar to torchvision.transforms.RandomResizedCrop
"""
from dataclasses import replace
from .utils import fast_crop
import numpy as np
from typing import Callable, Optional, Tuple
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.compiler import Compiler

class RandomResizedCrop(Operation):
    """Crop a random portion of image with random aspect ratio and resize it to
    a given size. Chances are you do not want to use this augmentation and
    instead want to include RRC as part of the decoder, by using the 
    :cla:`~ffcv.fields.rgb_image.ResizedCropRGBImageDecoder` class.

    Parameters
    ----------
    scale : Tuple[float, float]
        Lower and upper bounds for the ratio of random area of the crop.
    ratio : Tuple[float, float]
        Lower and upper bounds for random aspect ratio of the crop.
    size : int
        Side length of the output.
    """
    def __init__(self, scale: Tuple[float, float], ratio: Tuple[float, float], size: int):
        super().__init__()
        self.scale = scale
        self.ratio = ratio
        self.size = size
        

    def generate_code(self) -> Callable:
        scale, ratio = self.scale, self.ratio
        if isinstance(scale, tuple):
            scale = np.array(scale)
        if isinstance(ratio, tuple):
            ratio = np.array(ratio)
        my_range = Compiler.get_iterator()
        def random_resized_crop(images, dst):
            for idx in my_range(images.shape[0]):
                i, j, h, w = fast_crop.get_random_crop(images[idx].shape[0],
                                                    images[idx].shape[1],
                                                    scale,
                                                    ratio)
                fast_crop.resize_crop(images[idx], i, i + h, j, j + w, dst[idx])
            return dst

        random_resized_crop.is_parallel = True
        return random_resized_crop

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return replace(previous_state, jit_mode=True, shape=(self.size, self.size, 3)), \
               AllocationQuery((self.size, self.size, 3), dtype=previous_state.dtype)


