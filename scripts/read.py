from tqdm import tqdm
from torch.utils.data import Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, FloatField
from ffcv.reader import Reader
from ffcv.loader import Loader, OrderOption
from ffcv.memory_managers import RAMMemoryManager
from ffcv.transforms import Cutout, Collate, ToTensor
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt

if __name__ == '__main__':
    loader = Loader('/tmp/cifar_test_jpg.beton',
                    batch_size=128,
                    order=OrderOption.RANDOM)
    loader.pipelines['image'] = [
        Cutout(8),
        Collate(),
        ToTensor()
    ]
    print("Ratio raw", loader.reader.metadata['f0']['mode'].mean())

    for i in range(1):
        for image, label in tqdm(loader):
            pass
