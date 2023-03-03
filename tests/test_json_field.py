import string
from ctypes import pointer
from tempfile import NamedTemporaryFile
from collections import defaultdict
from multiprocessing import cpu_count

from assertpy import assert_that
import numpy as np
from torch.utils.data import Dataset
from ffcv import DatasetWriter
from ffcv.fields import IntField, JSONField
from ffcv.fields.bytes import BytesDecoder
from ffcv.fields.basics import IntDecoder
from ffcv import Loader

options = list(string.ascii_uppercase + string.digits)

def generate_random_string(low, high):
    length = np.random.randint(low, high)
    content = ''.join(np.random.choice(options, size=length))
    return content

class DummyDictDataset(Dataset):

    def __init__(self, n_samples):
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if index >= self.n_samples:
            raise IndexError()
        np.random.seed(index)
        length = np.random.randint(5, 250)
        content = np.random.randint(0, 256, size=(length,))
        json_content = {}
        for i in range(3):
            json_content[generate_random_string(5, 10)] = generate_random_string(50, 250)
        return index, json_content

def run_test(n_samples):
    with NamedTemporaryFile() as handle:
        name = handle.name
        dataset = DummyDictDataset(n_samples)
        writer = DatasetWriter(name, {
            'index': IntField(),
            'activations': JSONField()
        }, num_workers=min(3, cpu_count()))

        writer.from_indexed_dataset(dataset)

        loader = Loader(name, batch_size=3, num_workers=min(5, cpu_count()),
                        pipelines={
                            'activations': [BytesDecoder()],
                            'index': [IntDecoder()]
                        }
        )
        ix = 0
        for _, json_encoded in loader:
            json_docs = JSONField.unpack(json_encoded)
            for doc in json_docs:
                ref_doc = dataset[ix][1]
                assert_that(sorted(doc.items())).is_equal_to(sorted(ref_doc.items()))
                ix += 1


def test_simple_dict():
    run_test(32)

if __name__ == '__main__':
    test_simple_dict()