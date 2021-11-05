from typing import Sequence, TYPE_CHECKING
import numpy as np

from torch.utils.data import DistributedSampler

from .base import TraversalOrder

if TYPE_CHECKING:
    from ..loader.loader import Loader
    

class QuasiRandom(TraversalOrder):
    
    def __init__(self, loader:'Loader'):
        super().__init__(loader)
        
        # TODO filter only the samples we care about!!
        self.page_to_samples = loader.memory_manager.page_to_samples
        
        if not self.page_to_samples:
            raise ValueError("Dataset won't benefit from QuasiRandom order, use regular Random")
        
        if self.distributed:
            raise NotImplementedError("distributed Not implemented yet for QuasiRandom")
        

    def sample_order(self, epoch: int) -> Sequence[int]:
        generator = np.random.default_rng(self.seed + epoch if self.seed is not None else None)
        page_order = list(generator.permutation(list(self.page_to_samples.keys())))
        
        pages_in_reservoir = {}
        
        samples_available = 0
        
        result_order = []
        
        while len(result_order) < len(self.indices):
            # First we make sure the reserver is big enough
            while len(pages_in_reservoir) < 2 * self.loader.batch_size and page_order:
                current_page = page_order.pop()
                page_content = list(self.page_to_samples[current_page])
                generator.shuffle(page_content)
                pages_in_reservoir[current_page] = page_content
                samples_available += len(page_content)
                
            selected_page = generator.choice(list(pages_in_reservoir.keys()))
            page_content = pages_in_reservoir[selected_page]
            element = page_content.pop()
            result_order.append(element)
            if not page_content:
                del pages_in_reservoir[selected_page]
                
        return np.array(result_order)