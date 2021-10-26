import ctypes
from numba import njit
import numpy as np
from ctypes import CDLL, c_int64, c_uint8, c_uint64, POINTER, c_void_p, c_uint32, c_bool
import ffcv._libffcv

lib = CDLL(ffcv._libffcv.__file__)

# ctypes_resize = lib.resize
# ctypes_resize.argtypes = 11 * [c_int64]
ctypes_resize = None

# Extract and define the interface of imdeocde
ctypes_imdecode = lib.imdecode
ctypes_imdecode.argtypes = [
    c_void_p, c_uint64, c_uint32, c_uint32, c_void_p, c_uint32, c_uint32,
    c_uint32, c_uint32, c_uint32, c_uint32, c_bool
]


def imdecode(source: np.ndarray, dst: np.ndarray,
             source_height: int, source_width: int,
             crop_height=None, crop_width=None,
             offset_x=0, offset_y=0, scale_factor_num=1, scale_factor_denom=1,
             do_flip=False):
    return ctypes_imdecode(source.ctypes.data, source.size,
                           source_height, source_width, dst.ctypes.data,
                           crop_height, crop_width, offset_x, offset_y, scale_factor_num, scale_factor_denom,
                           do_flip)
