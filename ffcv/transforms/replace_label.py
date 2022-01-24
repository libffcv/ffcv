"""
Replace label
"""
from typing import Tuple

import numpy as np
from dataclasses import replace
from typing import Callable, Optional, Tuple
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State
from ..pipeline.compiler import Compiler

class ReplaceLabel(Operation):
    """Replace label of specified images.

    Parameters
    ----------
    indices : Sequence[int]
        The indices of images to relabel.
    new_label : int
        The new label to assign.
    """

    def __init__(self, indices, new_label: int):
        super().__init__()
        self.indices = np.sort(indices)
        self.new_label = new_label

    def generate_code(self) -> Callable:

        to_change = self.indices
        new_label = self.new_label
        my_range = Compiler.get_iterator()

        def replace_label(labels, temp_array, indices):
            for i in my_range(labels.shape[0]):
                sample_ix = indices[i]
                position = np.searchsorted(to_change, sample_ix)
                if position < len(to_change) and to_change[position] == sample_ix:
                    labels[i] = new_label
            return labels

        replace_label.is_parallel = True
        replace_label.with_indices = True

        return replace_label

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), None)
