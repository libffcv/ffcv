from tempfile import NamedTemporaryFile
import torch as ch
from tqdm import tqdm

from assertpy import assert_that
import numpy as np
from torch.utils.data import Dataset
from ffcv import DatasetWriter
from ffcv.fields import IntField, NDArrayField
from ffcv import Loader
from ffcv.fields.basics import IntDecoder
from ffcv.fields.ndarray import NDArrayDecoder
from ffcv.loader.loader import OrderOption
from ffcv.transforms import ToDevice, ToTensor, Squeeze
import time

class DummyArrayDataset(Dataset):
    def __init__(self, n_samples, shape):
        self.n_samples = n_samples
        self.shape = shape
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if index >= self.n_samples:
            raise IndexError()
        np.random.seed(index)
        return (np.random.rand(50000) > 0.5).astype('bool'), np.random.rand(50000).astype('float32'), index

def test_cuda():
    n_samples, shape = (300000, (50000,))
    with NamedTemporaryFile() as handle:
        # name = handle.name
        # dataset = DummyArrayDataset(n_samples, shape)
        # writer = DatasetWriter(len(dataset), name, {
        #     'mask': NDArrayField(dtype=np.dtype('bool'), shape=(50_000,)),
        #     'targets': NDArrayField(dtype=np.dtype('float32'), shape=(50_000,)),
        #     'idx': IntField()
        # })

        # with writer:
        #     writer.write_pytorch_dataset(dataset, num_workers=10)

        loader = Loader(
            '/mnt/nfs/home/aiilyas/subpops_betons/train_confs_50.beton',
                # name,
                batch_size=2048,
                num_workers=10,
                order=OrderOption.QUASI_RANDOM,
                indices=np.arange(n_samples),
                drop_last=True,
                os_cache=True,
                pipelines={
                    'mask': [NDArrayDecoder(), ToTensor(), ToDevice(ch.device('cuda:0'), non_blocking=True)],
                    'targets': [NDArrayDecoder(), ToTensor(), ToDevice(ch.device('cuda:0'), non_blocking=True)],
                    'idx': [IntDecoder(), ToTensor(), Squeeze(), ToDevice(ch.device('cuda:0'), non_blocking=True)]
                })

        indexer = ch.rand(300000, 50000)
        res = ch.empty(2048, 50000).cuda()

        chungus = ch.randn(50000, 50000).cuda()
        w_saga = ch.randn(50000, 50000).cuda()
        for X_bool, targs, ixes in tqdm(loader):
            a_i = indexer[ixes].cuda(non_blocking=True)
            X = X_bool.float()
            X_bool.logical_not_()
            
            X -= 0.5
            X /= 0.5

            B = X_bool.sum(0)
            ch.mm(X, chungus, out=res)
            res -= targs

            res *= X_bool
            res -= a_i
            ch.mm(X.T, res, out=w_saga)

        