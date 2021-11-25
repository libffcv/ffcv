from tempfile import NamedTemporaryFile
import torch as ch
from tqdm import tqdm
import time

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

def run_experiment_cuda(weight, loader, sync=False):
    total = 0.
    for X_bool, _, __ in tqdm(loader):
        if sync:
            ch.cuda.synchronize()
            time.sleep(1)
        total += X_bool.float() @ weight

    return total

def run_cuda(weight, sync):
    n_samples, shape = (2048 * 10, (50000,))
    with NamedTemporaryFile() as handle:
        name = handle.name
        dataset = DummyArrayDataset(n_samples, shape)
        writer = DatasetWriter(len(dataset), name, {
            'mask': NDArrayField(dtype=np.dtype('bool'), shape=(50_000,)),
            'targets': NDArrayField(dtype=np.dtype('float32'), shape=(50_000,)),
            'idx': IntField()
        })

        with writer:
            writer.write_pytorch_dataset(dataset, num_workers=10)

        loader = Loader(
                name,
                batch_size=2048,
                num_workers=10,
                order=OrderOption.QUASI_RANDOM,
                indices=np.arange(n_samples),
                drop_last=False,
                os_cache=True,
                pipelines={
                    'mask': [NDArrayDecoder(), ToTensor(), ToDevice(ch.device('cuda:0'), non_blocking=False)],
                    'targets': [NDArrayDecoder(), ToTensor(), ToDevice(ch.device('cuda:0'), non_blocking=False)],
                    'idx': [IntDecoder(), ToTensor(), Squeeze(), ToDevice(ch.device('cuda:0'), non_blocking=False)]
                })
        
        return run_experiment_cuda(weight, loader, sync)

weight = ch.randn(50_000, 50_000).cuda()
for t in [False, True, True]:
    print(run_cuda(weight, t))
