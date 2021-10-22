from tqdm import tqdm
from torch.utils.data import Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, FloatField
from ffcv.reader import Reader
from ffcv.loader import Loader, OrderOption
from ffcv.memory_managers import RAMMemoryManager
from ffcv.transforms import RandomResizedCrop, Cutout
import numpy as np

import matplotlib as mpl
mpl.use('module://imgcat')
from matplotlib import pyplot as plt

if __name__ == '__main__':
    loader = Loader('/tmp/test.beton',
                    batch_size=128,
                    order=OrderOption.RANDOM)
    loader.pipelines['image'] = [
        Cutout(8)
    ]
    
    for i in range(1):
        for image, label in tqdm(loader):
            print(image.shape)
            plt.imshow(image[0])
            plt.show()
            break
            pass
