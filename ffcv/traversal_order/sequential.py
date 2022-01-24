from typing import Sequence, TYPE_CHECKING
import numpy as np

from torch.utils.data import DistributedSampler

from .base import TraversalOrder

if TYPE_CHECKING:
    from ..loader.loader import Loader
    

class Sequential(TraversalOrder):
    
    def __init__(self, loader:'Loader'):
        super().__init__(loader)
        
        if self.distributed:
            self.sampler = DistributedSampler(self.indices,
                                              shuffle=False,
                                              seed=self.seed,
                                              drop_last=False)
        

    def sample_order(self, epoch: int) -> Sequence[int]:
        if not self.distributed:
            return self.indices
        
        self.sampler.set_epoch(epoch)
        
        return self.indices[np.array(list(self.sampler))]
