import numpy as np
import platform
from ctypes import CDLL, c_int64, c_uint8, c_uint64, POINTER, c_void_p, c_uint32, c_bool, cdll, c_char_p, c_int32, create_string_buffer
import ffcv._libffcv
import cv2

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
ctypes_resize.argtypes = 12 * [c_int64]

def resize_crop(source, start_row, end_row, start_col, end_col, destination,interpolation=3):
    ctypes_resize(0,
                  source.ctypes.data,
                  source.shape[0], source.shape[1],
                  start_row, end_row, start_col, end_col,
                  destination.ctypes.data,
                  destination.shape[0], destination.shape[1],interpolation)

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


# Extract and define the interface of imdeocde
ctypes_imcropresizedecode = lib.imcropresizedecode
ctypes_imcropresizedecode.argtypes = [
    c_void_p, c_uint64, 
    c_void_p, c_void_p, 
    c_uint32, c_uint32,
    c_uint32, c_uint32,
    c_uint32, c_uint32,
    c_uint32,
]

def imcropresizedecode(source: np.ndarray,  tmp: np.ndarray, dst: np.ndarray,
             crop_height: int, crop_width: int,
             offset_y=0, offset_x=0,
             interpolation=cv2.INTER_CUBIC):
    return ctypes_imcropresizedecode(
        source.ctypes.data, source.size, 
                tmp.ctypes.data, dst.ctypes.data,     
                dst.shape[0], dst.shape[1],           
                crop_height, crop_width, 
                offset_y, offset_x,
                interpolation)

ctypes_memcopy = lib.my_memcpy
ctypes_memcopy.argtypes = [c_void_p, c_void_p, c_uint64]

def memcpy(source: np.ndarray, dest: np.ndarray):
    return ctypes_memcopy(source.ctypes.data, dest.ctypes.data, source.size*source.itemsize)

ctypes_init_client = lib.init_client
ctypes_init_client.argtypes = [ c_char_p, c_int32]

def init_client(url:str = b"localhost", port:int = 12345):
    raise Exception("This function is not implemented. Because the multi-threading in numba will cause an error when reading the data. ")
    return ctypes_init_client(url, port)

ctypes_get_slice = lib.get_slice
ctypes_get_slice.argtypes = [c_int32, c_uint64, c_uint64, c_void_p]

def get_slice(sockfd:int,start: int, end: int,buffer: np.ndarray):
    raise Exception("This function is not implemented. Because the multi-threading in numba will cause an error when reading the data. ")
    return ctypes_get_slice(sockfd, start, end, buffer.ctypes.data)    
    

# ctypes_set_share_buffer = lib.set_share_buffer
# ctypes_set_share_buffer.argtypes = [c_void_p]

# def set_share_buffer(buffer: np.ndarray):
#     return ctypes_set_share_buffer(buffer.ctypes.data)

# ctypes_get_share_buffer = lib.get_share_buffer
# ctypes_get_slice.restype = c_int

# def get_share_buffer():    
#     return np.frombuffer(ctypes_get_share_buffer(),dtype=np.uint8)