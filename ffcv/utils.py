import numpy as np
from numba import types
from numba.extending import intrinsic


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
    