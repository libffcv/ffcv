"""
Random Erasing augmentation (https://arxiv.org/abs/1708.04896)
"""

# Implementation inspired by fastai https://docs.fast.ai/vision.augment.html#randomerasing
# fastai - Apache License 2.0 - Copyright (c) 2023 fast.ai

import math
import numpy as np
from numpy.random import rand
from typing import Callable, Optional, Tuple
from dataclasses import replace

from ..pipeline.compiler import Compiler
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State


class RandomErasing(Operation):
    """Random erasing data augmentation (https://arxiv.org/abs/1708.04896).

    Parameters
    ----------
    prob : float
        Probability of applying on each image.
    min_area : float
        Minimum erased area as percentage of image size.
    max_area : float
        Maximum erased area as percentage of image size.
    min_aspect : float
        Minimum aspect ratio of erased area.
    max_count : int
        Maximum number of erased blocks per image. Erased Area is scaled by max_count.
    fill_mean : Tuple[int, int, int], optional
        The RGB color mean (ImageNet's (124, 116, 103) by default) to randomly fill the
        erased area with. Should be the mean of dataset or pretrained dataset.
    fill_std : Tuple[int, int, int], optional
        The RGB color standard deviation (ImageNet's (58, 57, 57) by default) to randomly
        fill the erased area with. Should be the st. dev of dataset or pretrained dataset.
    fast_fill : bool
        Default of True is ~2X faster by generating noise once per batch and randomly
        selecting slices of the noise instead of generating unique noise per each image.
    """
    def __init__(self, prob: float, min_area: float = 0.02, max_area: float = 0.3,
                 min_aspect: float = 0.3, max_count: int = 1,
                 fill_mean: Tuple[int, int, int] = (124, 116, 103),
                 fill_std: Tuple[int, int, int] = (58, 57, 57),
                 fast_fill : bool = True):
        super().__init__()
        self.prob = np.clip(prob, 0., 1.)
        self.min_area = np.clip(min_area, 0., 1.)
        self.max_area = np.clip(max_area, 0., 1.)
        self.log_ratio = (math.log(np.clip(min_aspect, 0., 1.)), math.log(1/np.clip(min_aspect, 0., 1.)))
        self.max_count = max_count
        self.fill_mean = np.array(fill_mean)
        self.fill_std = np.array(fill_std)
        self.fast_fill = fast_fill

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        prob = self.prob
        min_area = self.min_area
        max_area = self.max_area
        log_ratio = self.log_ratio
        max_count = self.max_count
        fill_mean = self.fill_mean
        fill_std = self.fill_std
        fast_fill = self.fast_fill

        def random_erase(images, *_):
            if fast_fill:
                noise = fill_mean + (fill_std * np.random.randn(images.shape[1], images.shape[2], images.shape[3])).astype(images.dtype)

            should_cutout = rand(images.shape[0]) < prob
            for i in my_range(images.shape[0]):
                if should_cutout[i]:
                    count = np.random.randint(1, max_count) if max_count > 1 else 1
                    for j in range(count):
                        # Randomly select bounds
                        area = np.random.uniform(min_area, max_area, 1) * images.shape[1] * images.shape[2] / count
                        aspect = np.exp(np.random.uniform(log_ratio[0], log_ratio[1], 1))
                        bound = (
                            int(round(np.sqrt(area * aspect).item())),
                            int(round(np.sqrt(area / aspect).item())),
                        )
                        # Select random erased area
                        coord = (
                            np.random.randint(0, max(1, images.shape[1] - bound[0])),
                            np.random.randint(0, max(1, images.shape[2] - bound[1])),
                        )
                        # Fill image with random noise in-place
                        if fast_fill:
                            images[i, coord[0]:coord[0] + bound[0], coord[1]:coord[1] + bound[1]] =\
                                noise[coord[0]:coord[0] + bound[0], coord[1]:coord[1] + bound[1]]
                        else:
                            noise = fill_mean + (fill_std * np.random.randn(bound[0], bound[1], images.shape[3])).astype(images.dtype)
                            images[i, coord[0]:coord[0] + bound[0], coord[1]:coord[1] + bound[1]] = noise
            return images

        random_erase.is_parallel = True
        return random_erase

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return replace(previous_state, jit_mode=True), None