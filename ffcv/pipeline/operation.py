from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from typing import Callable, Optional, Tuple

import numpy as np

from .state import State
from .allocation_query import AllocationQuery

if TYPE_CHECKING:
    from ..fields.base import Field


class Operation(ABC):

    def __init__(self):
        self.metadata: np.ndarray = None
        self.memory_read: Callable[[np.uint64], np.ndarray] = None
        pass

    def accept_field(self, field: 'Field'):
        self.field: 'Field' = field

    def accept_globals(self, metadata, memory_read):
        self.metadata = metadata
        self.memory_read = memory_read

    # Return the code to run this operation
    @abstractmethod
    def generate_code(self) -> Callable:
        raise NotImplementedError

    def declare_shared_memory(self, previous_state: State) -> Optional[AllocationQuery]:
        return None

    def generate_code_for_shared_state(self) -> Optional[Callable]:
        return None

    @abstractmethod
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        raise NotImplementedError
