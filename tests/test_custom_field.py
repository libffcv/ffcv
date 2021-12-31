import pytest
import numpy as np
from uuid import uuid4
from ffcv.fields.ndarray import NDArrayField, NDArrayDecoder
from ffcv.writer import DatasetWriter
from ffcv.loader import Loader, OrderOption
from tempfile import NamedTemporaryFile

class StringDecoder(NDArrayDecoder):
    pass

class StringField(NDArrayField):
    def __init__(self, max_len: int, pad_char='\0'):
        self.max_len = max_len
        self.pad_char = pad_char
        super().__init__(np.dtype('uint8'), (max_len,))
    
    def encode(self, destination, field, malloc):
        padded_field = (field + self.pad_char * self.max_len)[:self.max_len]
        field = np.frombuffer(padded_field.encode('ascii'), dtype='uint8')
        return super().encode(destination, field, malloc)

MAX_STRING_SIZE = 100

class CaptionDataset:
    def __init__(self, N):
        self.captions = [str(uuid4())[:np.random.randint(50)] for _ in range(N)]

    def __getitem__(self, idx):
        return (self.captions[idx],)

    def __len__(self):
        return len(self.captions)

def test_string_field():
    dataset = CaptionDataset(100)

    with NamedTemporaryFile() as handle:
        writer = DatasetWriter(handle.name, {
            'label': StringField(MAX_STRING_SIZE)
        })

        writer.from_indexed_dataset(dataset)
        loader = Loader(handle.name,
                    batch_size=10,
                    num_workers=2,
                    order=OrderOption.RANDOM,
                    pipelines={
                        'label': [StringDecoder()]
                    },
                    custom_fields={
                        'label': StringField
                    })
        
        all_caps = []
        for x, in loader:
            for cap in x:
                all_caps.append(cap.tobytes().decode('ascii').replace('\0', ''))
        assert set(all_caps) == set(dataset.captions)

def test_no_custom_field():
    dataset = CaptionDataset(100)

    with NamedTemporaryFile() as handle:
        writer = DatasetWriter(handle.name, {
            'label': StringField(MAX_STRING_SIZE)
        })

        writer.from_indexed_dataset(dataset)
        with pytest.raises(ValueError):
            Loader(handle.name,
                    batch_size=10,
                    num_workers=2,
                    order=OrderOption.RANDOM,
                    pipelines={
                        'label': [StringDecoder()]
                    })
        