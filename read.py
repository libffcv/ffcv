from ffcv.loader.main_thread import OrderOption
from torch.utils.data import Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, FloatField
from ffcv.reader import Reader
from ffcv import Loader
from ffcv.memory_managers import RAMMemoryManager

if __name__ == '__main__':
    loader = Loader('/tmp/test.beton', order=OrderOption.RANDOM)
    
    for i in range(3):
        for _ in loader:
            pass