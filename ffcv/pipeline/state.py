
from dataclasses import dataclass
from typing import Literal, Tuple

import torch as ch
import numpy as np

@dataclass
class State:
    jit_mode: bool
    device: ch.device
    shape: Tuple[int, ...]
    dtype: np.dtype
    
    # Assess the validity of a pipeline stage
    def __post_init__(self):
        if self.jit_mode and self.device != ch.device('cpu'):
            raise AssertionError("Can't be in JIT mode and on the GPU")
        if self.jit_mode and isinstance(self.dtype, ch.dtype):
            raise AssertionError("Can't allocate a torch tensor in JIT mode")