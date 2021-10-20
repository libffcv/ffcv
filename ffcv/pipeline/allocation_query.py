from typing import Tuple
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AllocationQuery:
    shape: Tuple[int, ...] = (1,)
    dtype: np.dtype = np.dtype('<u1')