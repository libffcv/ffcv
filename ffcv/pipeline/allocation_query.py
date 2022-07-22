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

def allocate_query(memory_allocation: AllocationQuery, batch_size: int, batches_ahead: int):
    # We compute the total amount of memory needed for this
    # operation
    final_shape = [batches_ahead,
                   batch_size, *memory_allocation.shape]
    if isinstance(memory_allocation.dtype, ch.dtype):
        result = []
        for _ in range(final_shape[0]):
            partial = ch.empty(*final_shape[1:],
                              dtype=memory_allocation.dtype,
                              device=memory_allocation.device)
            try:
                partial = partial.pin_memory()
            except:
                pass
            result.append(partial)
    else:
        ch_dtype = ch.from_numpy(np.empty(0, dtype=memory_allocation.dtype)).dtype
        result = ch.empty(*final_shape,
                          dtype=ch_dtype)
        try:
            result = result.pin_memory()
        except:
            pass
        result = result.numpy()
    return result