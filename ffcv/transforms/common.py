from typing import Callable, Optional, Tuple
from ..pipeline.allocation_query import AllocationQuery
from ..pipeline.operation import Operation
from ..pipeline.state import State
from dataclasses import replace

class Squeeze(Operation):
    """Remove given dimensions of input of size 1.
    Operates on tensors.

    Parameters
    ----------
    *dims : List[int]
        Dimensions to squeeze.
    """

    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def generate_code(self) -> Callable:
        def squeeze(inp, _):
            inp.squeeze_(*self.dims)
            return inp
        return squeeze

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return replace(previous_state, shape=[x for x in previous_state.shape if not x == 1]), None
