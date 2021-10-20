from torch.utils.data import Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from torchvision.datasets import CIFAR10

if __name__ == '__main__':

    my_dataset = CIFAR10(root="./data")

    writer = DatasetWriter(len(my_dataset), '/tmp/test.beton', {
        'image': RGBImageField(write_mode='smart', smart_factor=2, max_resolution=16, smart_threshold=800),
        'label': IntField(),
    })

    with writer:
        writer.write_pytorch_dataset(my_dataset, num_workers=16, chunksize=100)
