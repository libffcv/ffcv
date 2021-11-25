from typing import Optional, Sequence, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch as ch


@dataclass(frozen=True)
class AllocationQuery:
    shape: Tuple[int, ...]
    dtype: Union[np.dtype, ch.dtype]
    device: Optional[ch.device] = None


Allocation = Union[AllocationQuery, Sequence[AllocationQuery]]