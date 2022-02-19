import numpy as np
from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.operation import Operation, AllocationQuery
from dataclasses import replace
from typing import Callable, Optional, Tuple
from ffcv.pipeline.state import State
from ffcv.transforms.utils.fast_crop import rotate, shear, blend, \
    adjust_contrast, posterize, invert, solarize, equalize, fast_equalize, \
    autocontrast, sharpen, adjust_saturation, translate, adjust_brightness

class RandAugment(Operation):
    def __init__(self, 
                 num_ops: int = 2, 
                 magnitude: int = 9, 
                 num_magnitude_bins: int = 31):
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        num_bins = num_magnitude_bins
        # index, name (for readability); bins, sign multiplier
        # those with a -1 can have negative magnitude with probability 0.5
        self.op_table = [
            (0, "Identity", np.array(0.0), 1),
            (1, "ShearX", np.linspace(0.0, 0.3, num_bins), -1),
            (2, "ShearY", np.linspace(0.0, 0.3, num_bins), -1),
            (3, "TranslateX", np.linspace(0.0, 150.0 / 331.0, num_bins), -1),
            (4, "TranslateY", np.linspace(0.0, 150.0 / 331.0, num_bins), -1),
            (5, "Rotate", np.linspace(0.0, 30.0, num_bins), -1),
            (6, "Brightness", np.linspace(0.0, 0.9, num_bins), -1),
            (7, "Color", np.linspace(0.0, 0.9, num_bins), -1),
            (8, "Contrast", np.linspace(0.0, 0.9, num_bins), -1),
            (9, "Sharpness", np.linspace(0.0, 0.9, num_bins), -1),
            (10, "Posterize", 8 - (np.arange(num_bins) / ((num_bins - 1) / 4)).round(), 1),
            (11, "Solarize", np.linspace(255.0, 0.0, num_bins), 1),
            (12, "AutoContrast", np.array(0.0), 1),
            (13, "Equalize", np.array(0.0), 1),
        ]

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        op_table = self.op_table
        magnitudes = np.array([(op[2][self.magnitude] if op[2].ndim > 0 else 0) for op in self.op_table])
        is_signed = np.array([op[3] for op in self.op_table])
        num_ops = self.num_ops
#         for i in range(len(magnitudes)):
#             print(i, op_table[i][1], '%.3f'%magnitudes[i])
        def randaug(im, mem):
            dst, scratch, lut, scratchf = mem
            for i in my_range(im.shape[0]):
                for n in range(num_ops):
                    if n == 0:
                        src = im
                    else:
                        src = dst
                        
                    idx = np.random.randint(low=0, high=13+1)
                    mag = magnitudes[idx]
                    if np.random.random() < 0.5:
                        mag = mag * is_signed[idx] 

                    # Not worth fighting numba at the moment.
                    # TODO
                    if idx == 0:
                        dst[i][:] = src[i]
                    
                    if idx == 1: # ShearX (0.004)
                        shear(src[i], dst[i], mag, 0)

                    if idx == 2: # ShearY
                        shear(src[i], dst[i], 0, mag)

                    if idx == 3: # TranslateX
                        translate(src[i], dst[i], int(src[i].shape[1] * mag), 0)

                    if idx == 4: # TranslateY
                        translate(src[i], dst[i], 0, int(src[i].shape[2] * mag))

                    if idx == 5: # Rotate
                        rotate(src[i], dst[i], mag)

                    if idx == 6: # Brightness
                        adjust_brightness(src[i], scratch[i][0], 1.0 + mag, dst[i])

                    if idx == 7: # Color
                        adjust_saturation(src[i], scratch[i][0], 1.0 + mag, dst[i])

                    if idx == 8: # Contrast
                        adjust_contrast(src[i], scratch[i][0], 1.0 + mag, dst[i])

                    if idx == 9: # Sharpness
                        sharpen(src[i], scratch[i][0], 1.0 + mag, dst[i])

                    if idx == 10: # Posterize
                        posterize(src[i], int(mag), dst[i])

                    if idx == 11: # Solarize
                        solarize(src[i], scratch[i][0], mag, dst[i])

                    if idx == 12: # AutoContrast (TODO: takes 0.04s -> 0.052s) (+0.01s)
                        autocontrast(src[i], scratchf[i][0], dst[i])
                    
                    if idx == 13: # Equalize (TODO: +0.008s)
                        equalize(src[i], lut[i], dst[i])
                
            return dst

        randaug.is_parallel = True
        return randaug

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        h, w, c = previous_state.shape
        return replace(previous_state, shape=previous_state.shape), [
            AllocationQuery(previous_state.shape, dtype=np.dtype('uint8')), 
            AllocationQuery((1, h, w, c), dtype=np.dtype('uint8')),
            AllocationQuery((c, 256), dtype=np.dtype('int16')),
            AllocationQuery((1, h, w, c), dtype=np.dtype('float32')),
        ]