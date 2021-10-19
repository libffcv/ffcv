from torch.utils.data import Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, FloatField
from ffcv.reader import Reader

if __name__ == '__main__':
    reader = Reader('/tmp/test.beton')

