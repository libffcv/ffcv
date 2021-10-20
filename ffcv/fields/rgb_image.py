from dataclasses import replace
from typing import Optional, Callable

import cv2
import numpy as np
from PIL.Image import Image

from .base import Field, ARG_TYPE
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.stage import Stage
from ..pipeline.allocation_query import AllocationQuery

IMAGE_MODES = {
    'jpg': 0,
    'raw': 1
}

def encode_jpeg(numpy_image):
    success, result = cv2.imencode('.jpg', numpy_image)

    if not success:
        raise ValueError("Impossible to encode image in jpeg")

    return result.reshape(-1)

def resizer(image, target_resolution):
    if target_resolution is None:
        return image
    original_size = np.array([image.shape[1], image.shape[0]])
    ratio = target_resolution / original_size.max()
    if ratio < 1:
        new_size = (ratio * original_size).astype(int)
        image = cv2.resize(image, tuple(new_size))
    return image


class RGBImageDecoder(Operation):

    def allocate_output(self) -> Optional[AllocationQuery]:
        AllocationQuery((256, 256, 3), np.dtype('<u1'))
        
    def advance_state(self, previous_state: State) -> State:
        return replace(previous_state, stage=Stage.INDIVIDUAL, jit_mode=True)
    
    def generate_code(self) -> Callable:
        def decode(field, memory, destination):
            return destination


class RGBImageField(Field):

    def __init__(self, write_mode='raw', smart_factor:float = None,
                 max_resolution: int=None, smart_threshold: int=None) -> None:
        self.write_mode = write_mode
        self.smart_factor = smart_factor
        self.smart_threshold = smart_threshold
        self.max_resolution = max_resolution

    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('mode', '<u1'),
            ('width', '<u2'),
            ('height', '<u2'),
            ('data_ptr', '<u8'),
        ])
        
    def get_decoder(self) -> Operation:
        return RGBImageDecoder()

    @staticmethod
    def from_binary(binary: ARG_TYPE) -> Field:
        return RGBImageField()

    def to_binary(self) -> ARG_TYPE:
        return np.zeros(1, dtype=ARG_TYPE)[0]

    def encode(self, destination, image, malloc):
        if isinstance(image, Image):
            image = np.array(image)

        if not isinstance(image, np.ndarray):
            raise TypeError(f"Unsupported image type {type(image)}")

        if image.dtype != np.uint8:
            raise ValueError("Image type has to be uint8")

        if image.shape[2] != 3:
            raise ValueError(f"Invalid shape for rgb image: {image.shape}")

        assert image.dtype == np.uint8

        image = resizer(image, self.max_resolution)

        write_mode = self.write_mode
        as_jpg = None

        if write_mode == 'smart':
            as_jpg = encode_jpeg(image)
            write_mode = 'raw'
            if self.smart_factor is not None:
                if as_jpg.nbytes * self.smart_factor <= image.nbytes:
                    write_mode = 'jpg'
            if self.smart_threshold is not None:
                if image.nbytes > self.smart_threshold:
                    write_mode = 'jpg'

        destination['mode'] = IMAGE_MODES[write_mode]
        destination['height'], destination['width'] = image.shape[:2]

        if write_mode == 'jpg':
            if as_jpg is None:
                as_jpg = encode_jpeg(image)
            destination['data_ptr'], storage = malloc(as_jpg.nbytes)
            storage[:] = as_jpg
        elif write_mode == 'raw':
            image_bytes = np.ascontiguousarray(image).view('<u1').reshape(-1)
            destination['data_ptr'], storage = malloc(image.nbytes)
            storage[:] = image_bytes
        else:
            raise ValueError(f"Unsupported write mode {self.write_mode}")
