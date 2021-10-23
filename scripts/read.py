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
mpl.use('module://imgcat')
from matplotlib import pyplot as plt

if __name__ == '__main__':
    loader = Loader('../imagenet_train_tiny.beton',
                    batch_size=100,
                    order=OrderOption.RANDOM)
    loader.pipelines['image'] = [
        RandomResizedCrop((0.08, 1.0), np.array([2/3., 4/3.]), 224),
        Collate(),
        ToTensor()
    ]
    print("Ratio raw", loader.reader.metadata['f0']['mode'].mean())

    for i in range(1):
        for image, label in tqdm(loader):
            """
            for j in range(4):
                plt.imshow(image[j])
                plt.show()
                plt.cla()
            plt.show()
            plt.cla()
            break
            """
