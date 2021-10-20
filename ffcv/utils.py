import numpy as np

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