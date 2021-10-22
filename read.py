from tqdm import tqdm
from torch.utils.data import Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, FloatField
from ffcv.reader import Reader
from ffcv.loader import Loader, OrderOption
from ffcv.memory_managers import RAMMemoryManager
from ffcv.transforms import Cutout, Collate
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt

if __name__ == '__main__':
    loader = Loader('/tmp/test.beton',
                    batch_size=128,
                    order=OrderOption.RANDOM)
    loader.pipelines['image'] = [
        Cutout(8),
        Collate()
    ]

    for i in range(1):
        for image, label in tqdm(loader):
            print(image.shape)
