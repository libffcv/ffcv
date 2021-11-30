from os import path

import numpy as np
import torch as ch
from torch.utils.data import Dataset
import webdataset as wds



class DummyDataset(Dataset):

    def __init__(self, l):
        self.l = l

    def __len__(self):
        return self.l

    def __getitem__(self, index):
        if index > self.l:
            raise IndexError()
        return (index, np.sin(index))

def write_webdataset(folder, dataset):
    pattern = path.join(folder, "dataset-%06d.tar")
    writer = wds.TarWriter(pattern, maxcount=10)
    with writer as sink:
        for i, sample in enumerate(dataset):
            data = {
                '__key__': f'sample_{i}'
            }
            for j, value in enumerate(sample):
                data[f'data_{j}'] = value
            sink.write(data)


if __name__ == '__main__':

