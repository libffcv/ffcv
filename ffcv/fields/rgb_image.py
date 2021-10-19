import cv2
import numpy as np
from PIL.Image import Image
import io

from .base import Field, ARG_TYPE

IMAGE_MODES = {
    'jpg': 0,
    'raw': 1
}

def encode_jpeg(numpy_image):
    success, result = cv2.imencode('.jpg', numpy_image)

    if not success:
        raise ValueError("Impossible to encode image in jpeg")

    return result.reshape(-1)

class RGBImageField(Field):

    def __init__(self, write_mode='raw', smart_factor=2) -> None:
        self.write_mode = write_mode
        self.smart_factor = smart_factor

    @property
    def metadata_type(self) -> np.dtype:
        return np.dtype([
            ('mode', '<u1'),
            ('width', '<u2'),
            ('height', '<u2'),
            ('data_ptr', '<u8'),
        ])

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

        write_mode = self.write_mode
        as_jpg = None

        if write_mode == 'smart':
            as_jpg = encode_jpeg(image)
            if as_jpg.nbytes * self.smart_factor > image.nbytes:
                write_mode = 'raw'
            else:
                write_mode = 'jpg'

        destination['mode'] = IMAGE_MODES[write_mode]
        destination['height'], destination['width'] = image.shape[:2]

        print('using', write_mode)
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
