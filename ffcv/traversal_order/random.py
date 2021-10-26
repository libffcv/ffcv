from typing import Sequence

import numpy as np

from .base import TraversalOrder

class Random(TraversalOrder):

    def sample_order(self, epoch: int) -> Sequence[int]:
        generator = np.random.default_rng(self.seed + epoch if self.seed is not None else None)
        return generator.permutation(self.indices)