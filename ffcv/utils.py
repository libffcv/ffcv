import numpy as np
from numba import types
from numba.extending import intrinsic
import PIL.Image as Image


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def is_power_of_2(n):
    return (n & (n-1) == 0) and n != 0

def align_to_page(ptr, page_size):
    # If we are not aligned with the start of a page:
    if ptr % page_size != 0:
        ptr = ptr  + page_size - ptr % page_size
    return ptr

def decode_null_terminated_string(bytes: np.ndarray):
    return bytes.tobytes().decode('ascii').split('\x00')[0]

@intrinsic
def cast_int_to_byte_ptr(typingctx, src):
    # check for accepted types
    if isinstance(src, types.Integer):
        # create the expected type signature
        result_type = types.CPointer(types.uint8)
        sig = result_type(types.uintp)
        # defines the custom code generation
        def codegen(context, builder, signature, args):
            # llvm IRBuilder code here
            [src] = args
            rtype = signature.return_type
            llrtype = context.get_value_type(rtype)
            return builder.inttoptr(src, llrtype)
        return sig, codegen

from threading import Lock
s_print_lock = Lock()


def s_print(*a, **b):
    """Thread safe print function"""
    with s_print_lock:
        print(*a, **b)


# From https://uploadcare.com/blog/fast-import-of-pillow-images-to-numpy-opencv-arrays/
# Up to 2.5 times faster with the same functionality and a smaller number of allocations than numpy.asarray(img)
def pil_to_numpy(img:Image.Image) -> np.ndarray:
    "Fast conversion of Pillow `Image` to NumPy NDArray"
    img.load()
    # unpack data
    enc = Image._getencoder(img.mode, 'raw', img.mode)
    enc.setimage(img.im)

    # NumPy buffer for the result
    shape, typestr = Image._conv_type_shape(img)
    data = np.empty(shape, dtype=np.dtype(typestr))
    mem = data.data.cast('B', (data.data.nbytes,))

    bufsize, s, offset = 65536, 0, 0
    while not s:
        l, s, d = enc.encode(bufsize)
        mem[offset:offset + len(d)] = d
        offset += len(d)
    if s < 0:
        raise RuntimeError("encoder error %d in tobytes" % s)
    return data