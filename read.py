from tqdm import tqdm
from torch.utils.data import Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, FloatField
from ffcv.reader import Reader
from ffcv.loader import Loader, OrderOption
from ffcv.memory_managers import RAMMemoryManager

if __name__ == '__main__':
    loader = Loader('/tmp/test.beton',
                    batch_size=128,
                    order=OrderOption.RANDOM)
    loader.pipelines['image'] = [

    ]
    
    for i in range(2):
        for image, label in tqdm(loader):
            pass
