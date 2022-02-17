import ctypes
from numba import njit, prange
import numpy as np
from ...libffcv import ctypes_resize, ctypes_rotate, ctypes_shear, \
    ctypes_add_weighted, ctypes_equalize, ctypes_unsharp_mask

"""
Requires a float32 scratch array
"""
@njit(parallel=True, fastmath=True, inline='always')
def autocontrast(source, scratchf, destination):
    # numba: no kwargs in min? as a consequence, I might as well have written
    # this in C++
    # TODO assuming 3 channels
    minimum = [source[..., 0].min(), source[..., 1].min(), source[..., 2].min()]
    maximum = [source[..., 0].max(), source[..., 1].max(), source[..., 2].max()]
    scale = [0.0, 0.0, 0.0]
    for i in prange(source.shape[-1]):
        if minimum[i] == maximum[i]:
            scale[i] = 1
            minimum[i] = 0
        else:
            scale[i] = 255. / (maximum[i] - minimum[i])
    for i in prange(source.shape[-1]): 
        scratchf[..., i] = source[..., i] - minimum[i]
        scratchf[..., i] = scratchf[..., i] * scale[i]
    np.clip(scratchf, 0, 255, out=scratchf)
    destination[:] = scratchf


"""
Custom equalize -- equivalent to torchvision.transforms.functional.equalize,
but probably slow -- scratch is a (channels, 256) uint16 array.
"""
@njit(parallel=True, fastmath=True, inline='always')
def equalize(source, scratch, destination):
    for i in prange(source.shape[-1]):
        # TODO memory less than ideal for bincount() and hist()
        scratch[i] = np.bincount(source[..., i].flatten(), minlength=256)
        nonzero_hist = scratch[i][scratch[i] != 0]
        step = nonzero_hist[:-1].sum() // 255
    
        if step == 0:
            continue
        
        scratch[i][1:] = scratch[i].cumsum()[:-1]
        scratch[i] = (scratch[i] + step // 2) // step
        scratch[i][0] = 0
        np.clip(scratch[i], 0, 255, out=scratch[i])
        
        # numba doesn't like 2d advanced indexing
        for row in prange(source.shape[0]):
            destination[row, :, i] = scratch[i][source[row, :, i]]

"""
Equalize using OpenCV -- not equivalent to
torchvision.transforms.functional.equalize for so-far-unknown reasons.
"""
@njit(parallel=False, fastmath=True, inline='always')
def fast_equalize(source, chw_scratch, destination):
    # this seems kind of hacky
    # also, assuming ctypes_equalize allocates a minimal amount of memory
    # which may be incorrect -- so maybe we should do this from scratch.
    # TODO may be a better way to do this in pure OpenCV
    c, h, w = chw_scratch.shape
    chw_scratch[0] = source[..., 0]
    ctypes_equalize(chw_scratch.ctypes.data,
                    chw_scratch.ctypes.data,
                    h, w)
    chw_scratch[1] = source[..., 1]
    ctypes_equalize(chw_scratch.ctypes.data + h*w,
                    chw_scratch.ctypes.data + h*w,
                    h, w)
    chw_scratch[2] = source[..., 2]
    ctypes_equalize(chw_scratch.ctypes.data + 2*h*w,
                    chw_scratch.ctypes.data + 2*h*w,
                    h, w)
    destination[..., 0] = chw_scratch[0]
    destination[..., 1] = chw_scratch[1]
    destination[..., 2] = chw_scratch[2]


@njit(parallel=False, fastmath=True, inline='always')
def invert(source, destination):
    destination[:] = 255 - source


@njit(parallel=False, fastmath=True, inline='always')
def solarize(source, threshold, destination):
    invert(source, destination)
    destination[:] = np.where(source >= threshold, destination, source)


@njit(parallel=False, fastmath=True, inline='always')
def posterize(source, bits, destination):
    mask = ~(2 ** (8 - bits) - 1)
    destination[:] = source & mask
    
    
@njit(inline='always')
def blend(source1, source2, ratio, destination):
    ctypes_add_weighted(source1.ctypes.data, ratio,
                        source2.ctypes.data, 1 - ratio,
                        destination.ctypes.data,
                        source1.shape[0], source1.shape[1])

    
@njit(parallel=False, fastmath=True, inline='always')
def adjust_saturation(source, scratch, factor, destination):
    # TODO numpy autocasting probably allocates memory here,
    # should be more careful.
    # TODO do we really need scratch for this? could use destination
    scratch[...,0] = 0.299 * source[..., 0] + \
                     0.587 * source[..., 1] + \
                     0.114 * source[..., 2]
    scratch[...,1] = scratch[...,0]
    scratch[...,2] = scratch[...,1]
    
    blend(source, scratch, factor, destination)


@njit(parallel=False, fastmath=True, inline='always')
def adjust_contrast(source, scratch, factor, destination):
    # TODO assuming 3 channels
    scratch[:,:,:] = np.mean(0.299 * source[..., 0] +
                             0.587 * source[..., 1] + 
                             0.114 * source[..., 2])
    
    blend(source, scratch, factor, destination)


@njit(fastmath=True, inline='always')
def sharpen(source, destination, amount):
    ctypes_unsharp_mask(source.ctypes.data, 
                  destination.ctypes.data, 
                  source.shape[0], source.shape[1])
    
    # in PyTorch's implementation,
    # the border is unaffected
    destination[0,:] = source[0,:]
    destination[1:,0] = source[1:,0]
    destination[-1,:] = source[-1,:]
    destination[1:-1,-1] = source[1:-1,-1]
    
    blend(source, destination, amount, destination)


"""
Translation, x and y
Assuming this is faster than warpAffine;
also assuming tx and ty are ints
"""
@njit(inline='always')
def translate(source, destination, tx, ty):
    if tx > 0:
        destination[:, tx:] = source[:, :-tx]
        destination[:, :tx] = 0
    if tx < 0:
        destination[:, :tx] = source[:, -tx:]
        destination[:, tx:] = 0
    if ty > 0:
        destination[ty:, :] = source[:-ty, :]
        destination[:ty, :] = 0
    if ty < 0:
        destination[:ty, :] = source[-ty:, :]
        destination[ty:, :] = 0


@njit(inline='always')
def rotate(source, destination, angle):
    ctypes_rotate(angle, 
                  source.ctypes.data, 
                  destination.ctypes.data, 
                  source.shape[0], source.shape[1])


@njit(inline='always')
def shear(source, destination, shear_x, shear_y):
    ctypes_shear(shear_x, shear_y, 
                 source.ctypes.data, 
                 destination.ctypes.data, 
                 source.shape[0], source.shape[1])


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