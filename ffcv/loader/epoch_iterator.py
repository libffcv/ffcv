from threading import Thread
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .loader import Loader

class EpochIterator(Thread):

    def __init__(self, loader: 'Loader', epoch: int, order:Sequence[int]):
        self.loader: 'loader' = loader
        self.order = order
        self.idx_iter = iter(order)
        for name, handler in self.loader.reader.handlers.items():
            print(name)

        print(self.loader.pipelines['image'])
        
    def __next__(self):
        raise StopIteration()