import ctypes
from numba import njit
import numpy as np
from ...libffcv import ctypes_resize

@njit(inline='always')
def resize_crop(source, start_row, end_row, start_col, end_col, destination):
    ctypes_resize(0,
                  source.ctypes.data,
                  source.shape[0], source.shape[1],
                  start_row, end_row, start_col, end_col,
                  destination.ctypes.data,
                  destination.shape[0], destination.shape[1])


@njit(parallel=False, fastmath=True, inline='always')
def get_random_crop(height, width, scale, ratio):
    area = height * width
    log_ratio = np.log(ratio)
    for _ in range(10):
        target_area = area * np.random.uniform(scale[0], scale[1])
        aspect_ratio = np.exp(np.random.uniform(log_ratio[0], log_ratio[1]))
        w = int(round(np.sqrt(target_area * aspect_ratio)))
        h = int(round(np.sqrt(target_area / aspect_ratio)))
        if 0 < w <= width and 0 < h <= height:
            i = int(np.random.uniform(0, height - h + 1))
            j = int(np.random.uniform(0, width - w + 1))
            return i, j, h, w
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


@njit(parallel=False, fastmath=True, inline='always')
def get_center_crop(height, width, ratio):
    s = min(height, width)
    c = int(ratio * s)
    delta_h = (height - c) // 2
    delta_w = (width - c) // 2

    return delta_h, delta_w, c, c
