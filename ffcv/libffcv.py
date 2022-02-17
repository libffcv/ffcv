import ctypes
from numba import njit
import numpy as np
import platform
from ctypes import CDLL, c_int64, c_uint8, c_uint64, POINTER, c_void_p, c_uint32, c_bool, cdll
import ffcv._libffcv

lib = CDLL(ffcv._libffcv.__file__)
if platform.system() == "Windows":
    libc = cdll.msvcrt
    read_c = libc._read
else:
    libc = cdll.LoadLibrary('libc.so.6')
    read_c = libc.pread

read_c.argtypes = [c_uint32, c_void_p, c_uint64, c_uint64]

def read(fileno:int, destination:np.ndarray, offset:int):
    return read_c(fileno, destination.ctypes.data, destination.size, offset)


ctypes_resize = lib.resize
ctypes_resize.argtypes = 11 * [c_int64]

def resize_crop(source, start_row, end_row, start_col, end_col, destination):
    ctypes_resize(0,
                  source.ctypes.data,
                  source.shape[0], source.shape[1],
                  start_row, end_row, start_col, end_col,
                  destination.ctypes.data,
                  destination.shape[0], destination.shape[1])

# Extract and define the interface of imdeocde
ctypes_imdecode = lib.imdecode
ctypes_imdecode.argtypes = [
    c_void_p, c_uint64, c_uint32, c_uint32, c_void_p, c_uint32, c_uint32,
    c_uint32, c_uint32, c_uint32, c_uint32, c_bool, c_bool
]

def imdecode(source: np.ndarray, dst: np.ndarray,
             source_height: int, source_width: int,
             crop_height=None, crop_width=None,
             offset_x=0, offset_y=0, scale_factor_num=1, scale_factor_denom=1,
             enable_crop=False, do_flip=False):
    return ctypes_imdecode(source.ctypes.data, source.size,
                           source_height, source_width, dst.ctypes.data,
                           crop_height, crop_width, offset_x, offset_y, scale_factor_num, scale_factor_denom,
                           enable_crop, do_flip)


ctypes_memcopy = lib.my_memcpy
ctypes_memcopy.argtypes = [c_void_p, c_void_p, c_uint64]

def memcpy(source: np.ndarray, dest: np.ndarray):
    return ctypes_memcopy(source.ctypes.data, dest.ctypes.data, source.size*source.itemsize)

