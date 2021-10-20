from abc import ABC, abstractmethod
from typing import Sequence
from ..reader import Reader

class TraversalOrder(ABC):
    
    def __init__(self, indices: Sequence[int], reader: Reader, seed: int=42):
        self.indices = indices
        self.reader = reader
        self.seed = seed
        
    @abstractmethod
    def sample_order(self, epoch:int) -> Sequence[int]:
        raise NotImplemented()