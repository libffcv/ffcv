from torch.utils.data import Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from torchvision.datasets import ImageFolder

if __name__ == '__main__':

    my_dataset = ImageFolder(root="/mnt/nfs/datasets/imagenet/train")

    writer = DatasetWriter(len(my_dataset), '/tmp/imagenet_train.beton', {
        'image': RGBImageField(write_mode='smart', smart_factor=2, max_resolution=1024, smart_threshold=500),
        'label': IntField(),
    })

    with writer:
        writer.write_pytorch_dataset(my_dataset, num_workers=16, chunksize=100)
