import ctypes
from numba import njit
import numpy as np
from ctypes import CDLL, c_int64, c_uint8, c_uint64, POINTER
import ffcv._libffcv

lib = CDLL(ffcv._libffcv.__file__)

ctypes_resize = lib.resize
ctypes_resize.argtypes = 11 * [c_int64]

# Extract and define the interface of imdeocde
ctypes_imdecode = lib.imdecode
ctypes_imdecode.argtypes = [
    c_uint64,
    c_uint64,
    c_uint64,
    c_uint64, c_uint64
]


def imdecode(source: np.ndarray, dst: np.ndarray):
    return ctypes_imdecode(source.ctypes.data, source.size,
                           dst.ctypes.data, dst.shape[0], dst.shape[1])
