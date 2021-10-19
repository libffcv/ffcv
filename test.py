from torch.utils.data import Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from torchvision.datasets import ImageFolder, CIFAR10

class TestDataset(Dataset):

    def __init__(self):
        super().__init__()

    def __len__(self):
        return 100

    def __getitem__(self, index):
        return (index, 3.2, 42, -0.66)

if __name__ == '__main__':

    my_dataset = CIFAR10(root="./data")
    writer = DatasetWriter(len(my_dataset), '/tmp/test.beton', (
        RGBImageField(write_mode='smart', smart_factor=1),
        IntField(),
    ))
    with writer:
        writer.write_pytorch_dataset(my_dataset, num_workers=16, chunksize=100)
    print("done")