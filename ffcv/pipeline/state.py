
from dataclasses import dataclass
from typing import Literal, Tuple

import torch as ch

from .stage import Stage

@dataclass
class State:
    stage: Literal[Stage.INDIVIDUAL,
                   Stage.BATCHES]
    jit_mode: bool
    device: ch.device
    shape: Tuple[int, ...]
    random_seed: int
    
    # Assess the validity of a pipeline stage
    def __post_init__(self):
        if self.jit_mode and self.device != ch.device('cpu'):
            raise AssertionError("Can't be in JIT mode and on the GPU")
        
        if self.stage == Stage.INDIVIDUAL and not self.jit_mode:
            raise AssertionError("Individual processing has to be in JIT mode")
        