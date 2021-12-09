import numpy as np
from uuid import uuid4
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder
from tempfile import NamedTemporaryFile

MAX_STRING_SIZE = 100

class CaptionDataset:
    def __init__(self, N):
        self.captions = [str(uuid4())[:np.random.randint(50)] for _ in range(N)]

    def __getitem__(self, idx):
        padded_caption = (self.captions[idx] + (" " * MAX_STRING_SIZE))[:MAX_STRING_SIZE]
        return (np.frombuffer(padded_caption.encode('ascii'), dtype='uint8'),)

    def __len__(self):
        return len(self.captions)

dataset = CaptionDataset(100)

with NamedTemporaryFile() as handle:
    writer = DatasetWriter(len(dataset), handle.name, {
        'label': NDArrayField(np.dtype('uint8'), (MAX_STRING_SIZE,))
    })

    with writer:
        writer.write_pytorch_dataset(dataset, num_workers=1)
    
    loader = Loader(handle.name,
                batch_size=10,
                num_workers=2,
                order=OrderOption.RANDOM,
                pipelines={
                    'label': [NDArrayDecoder()]
                })
    
    for x, in loader:
        for cap in x:
            print(cap.tobytes().decode('ascii').strip())
    
