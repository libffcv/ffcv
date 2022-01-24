from typing import Sequence

import numpy as np
from torch.utils.data import DistributedSampler

from .base import TraversalOrder

class Random(TraversalOrder):

    def __init__(self, loader:'Loader'):
        super().__init__(loader)

        if self.distributed:
            self.sampler = DistributedSampler(self.indices,
                                              shuffle=True,
                                              seed=self.seed,
                                              drop_last=False)


    def sample_order(self, epoch: int) -> Sequence[int]:
        if not self.distributed:
            generator = np.random.default_rng(self.seed + epoch if self.seed is not None else None)
            return generator.permutation(self.indices)

        self.sampler.set_epoch(epoch)

        return self.indices[np.array(list(self.sampler))]
