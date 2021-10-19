from torch.utils.data import Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, FloatField
from ffcv.reader import Reader
from ffcv.memory_managers import RAMMemoryManager

if __name__ == '__main__':
    reader = Reader('/tmp/test.beton')
    manager = RAMMemoryManager(reader)

