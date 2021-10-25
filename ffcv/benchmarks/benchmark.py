from abc import ABCMeta, abstractmethod
from contextlib import AbstractContextManager

class Benchmark(AbstractContextManager, metaclass=ABCMeta):
    
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def run(self):
        raise NotImplemented()