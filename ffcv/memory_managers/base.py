from abc import abstractmethod, ABCMeta
from contextlib import AbstractContextManager


class MemoryManager(AbstractContextManager, metaclass=ABCMeta):

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        super().__init__()

    @abstractmethod
    def schedule_epoch(self, schedule):
        raise NotImplemented()

    @abstractmethod
    def read(self, address):
        raise NotImplemented()

    @abstractmethod
    def __enter__(self):
        return super().__enter__()

    @abstractmethod
    def __exit__(self, __exc_type, __exc_value, __traceback):
        return super().__exit__(__exc_type, __exc_value, __traceback)