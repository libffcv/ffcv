from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

from .state import State
from .allocation_query import AllocationQuery

class Operation(ABC):

    def __init__(self):
        pass
    
    # Return the code to run this operation
    @abstractmethod
    def generate_code(self) -> Callable:
        raise NotImplementedError
    
    @abstractmethod
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]: 
        raise NotImplementedError