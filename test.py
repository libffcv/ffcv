from multiprocessing.sharedctypes import Value
from torch.utils.data import Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
from torchvision.datasets import CIFAR10, ImageFolder

from argparse import ArgumentParser
from fastargs import Section, Param
from fastargs.validation import And, OneOf
from fastargs.decorators import param, section
from fastargs import get_current_config

Section('writer', 'arguments to give the writer').params(
    dataset=Param(And(str, OneOf(['cifar', 'imagenet'])), 'Which dataset to write', required=True),
    data_dir=Param(str, 'Where to find the PyTorch dataset', required=True),
    write_dir=Param(str, 'Where to write the new dataset', required=True),
    max_resolution=Param(int, 'Max image side length', required=True),
    smart_threshold=Param(int, 'Max image side length before compression', default=500),
    num_workers=Param(int, 'Number of workers to use', default=16),
    chunk_size=Param(int, 'Chunk size for writing', default=100)
)

@section('writer')
@param('dataset')
@param('data_dir')
@param('max_resolution')
@param('smart_threshold')
@param('num_workers')
@param('chunk_size')
def main(dataset, data_dir, max_resolution, smart_threshold, num_workers, chunk_size):
    if dataset == 'cifar':
        my_dataset = CIFAR10(root=data_dir, download=True)
    elif dataset == 'imagenet':
        my_dataset = ImageFolder(root=data_dir)
    else:
        raise ValueError('Unrecognized dataset', dataset)

    writer = DatasetWriter(len(my_dataset), '/tmp/test.beton', {
        'image': RGBImageField(write_mode='smart', 
                               smart_factor=2, 
                               max_resolution=max_resolution, 
                               smart_threshold=smart_threshold),
        'label': IntField(),
    })

    with writer:
        writer.write_pytorch_dataset(my_dataset, num_workers=num_workers, chunksize=100)

if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()