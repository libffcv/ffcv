from typing import Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch as ch

from ffcv.pipeline.stage import Stage


@dataclass(frozen=True)
class AllocationQuery:
    shape: Tuple[int, ...]
    dtype: Union[np.dtype, ch.dtype]
    device: Optional[ch.device] = None