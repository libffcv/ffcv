from tqdm import tqdm
from torch.utils.data import Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, FloatField
from ffcv.reader import Reader
from ffcv.loader import Loader, OrderOption
from ffcv.memory_managers import RAMMemoryManager
from ffcv.transforms import Cutout, Collate, ToTensor, RandomResizedCrop
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt

if __name__ == '__main__':
    loader = Loader('/tmp/imagenet_train.beton',
                    batch_size=512,
                    order=OrderOption.RANDOM)
    loader.pipelines['image'] = [
        # RandomResizedCrop((0.08, 1.0), np.array([2/3., 4/3.]), 224),
        Collate(),
        ToTensor()
    ]
    
    for i in range(2):
        for image, label in tqdm(loader):
            pass
