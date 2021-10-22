from typing import Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch as ch

from ffcv.pipeline.stage import Stage


@dataclass(frozen=True)
class AllocationQuery:
    shape: Tuple[int, ...] = (1,)
    dtype: Union[np.dtype, ch.dtype] = np.dtype('<u1')
    device: str = 'cpu'