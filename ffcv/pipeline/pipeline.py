from typing import Sequence
import torch as ch

from .state import State
from .stage import Stage
from .operation import Operation

class Pipeline:

    def __init__(self, operations: Sequence[Operation]):
        
        # This is the starting state of the pipeline
        self.original_state = State(stage=Stage.INDIVIDUAL,
                                    jit_mode=True,
                                    device=ch.device('cpu'),
                                    shape=None)
        
        current_state = self.original_state
        self.operations = operations
        
        self.memory_buffers = {}

        # For validation purposes
        for operation in operations:
            current_state = operation.advance_state(current_state)
            
    def allocate_memory(self):

        
        
    def compile(self):
        pass
