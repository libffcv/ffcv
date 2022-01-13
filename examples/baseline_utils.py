from torchvision import transforms
from torchvision import datasets
from fastargs.decorators import param
from fastargs import Param, Section

import torch
ch = torch

VAL_PATH = '/mnt/cfs/datasets/pytorch_imagenet/val'
TRAIN_PATH = '/mnt/cfs/datasets/pytorch_imagenet/train'

Section('baselines', 'baseline info').params(
    train_path=Param(str, 'path to train dataset', default=TRAIN_PATH),
    val_path=Param(str, 'path to val dataset', default=VAL_PATH),
    use_baseline=Param(int, '1 if yes 0 if no default 0', default=0)
)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

class GPULoader(ch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def __iter__(self, *args, **kwargs):
        iterator = super().__iter__(*args, **kwargs)
        gpuifier = lambda t: (t[0].cuda().half(), t[1].cuda())
        return map(gpuifier, iterator)

@param('baselines.train_path')
@param('data.num_workers')
@param('training.batch_size')
def baseline_train_loader(train_path, num_workers, batch_size):
    train_dataset = datasets.ImageFolder(
        train_path,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = GPULoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    return train_loader

@param('baselines.val_path')
@param('data.num_workers')
@param('validation.batch_size')
def baseline_val_loader(val_path, num_workers, batch_size):
    ds = datasets.ImageFolder(val_path, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ]))

    val_loader = GPULoader(ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)

    return val_loader