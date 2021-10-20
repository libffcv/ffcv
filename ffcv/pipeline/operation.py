from abc import ABC, abstractmethod
from typing import Callable, Optional

from .state import State
from .allocation_query import AllocationQuery

class Operation(ABC):

    def __init__(self):
        pass
    
    # Return the code to run this operation
    @abstractmethod
    def generate_code(self) -> Callable:
        raise NotImplemented()
    
    @abstractmethod
    def advance_state(self, previous_state: State) -> State: 
        raise NotImplemented()
    
    def allocate_output(self) -> Optional[AllocationQuery]:
        pass