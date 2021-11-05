from tempfile import NamedTemporaryFile
from assertpy.assertpy import assert_that

from assertpy import assert_that
import numpy as np
from torch.utils.data import Dataset
from ffcv import DatasetWriter
from ffcv.fields import IntField, NDArrayField
from ffcv import Loader

class DummyActivationsDataset(Dataset):

    def __init__(self, n_samples, shape):
        self.n_samples = n_samples
        self.shape = shape
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if index >= self.n_samples:
            raise IndexError()
        np.random.seed(index)
        return index, np.random.randn(*self.shape).astype('<f4')
    

def run_test(n_samples, shape):
    with NamedTemporaryFile() as handle:
        name = handle.name
        dataset = DummyActivationsDataset(n_samples, shape)
        writer = DatasetWriter(n_samples, name, {
            'index': IntField(),
            'activations': NDArrayField(np.dtype('<f4'), shape)
        })

        with writer:
            writer.write_pytorch_dataset(dataset, num_workers=3)
            
        import time
            
        loader = Loader(name, batch_size=3, num_workers=5)
        for ixes, activations in loader:
            for ix, activation in zip(ixes, activations):
                assert_that(np.all(dataset[ix][1] == activation.numpy())).is_true()
    
    
def test_simple_activations():
    run_test(4096, (2048,))
