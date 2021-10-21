from threading import Thread
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .loader import Loader

class EpochIterator(Thread):

    def __init__(self, loader: 'Loader', epoch: int, order:Sequence[int]):
        self.loader: 'loader' = loader
        self.order = order
        self.idx_iter = iter(order)
        self.allocate_memory()
        
    def allocate_memory(self):
        for name in self.loader.reader.handlers:
            self.loader.pipelines[name].allocate_memory(self.loader.batch_size)
        
    def __next__(self):
        raise StopIteration()