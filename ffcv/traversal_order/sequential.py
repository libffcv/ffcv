from typing import Sequence
from .base import TraversalOrder

class Sequential(TraversalOrder):

    def sample_order(self, epoch: int) -> Sequence[int]:
        return self.indices