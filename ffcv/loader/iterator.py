from threading import Thread
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .main_thread import Loader

class EpochIterator(Thread):

    def __init__(self, loader: 'Loader', epoch: int, order:Sequence[int]):
        self.loader = loader
        self.order = order
        
    def __next__(self):
        raise StopIteration()