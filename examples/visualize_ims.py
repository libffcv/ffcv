import os
import torch as ch
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from argparse import ArgumentParser

from matplotlib import pyplot as plt

def create_train_loader(train_dataset, batch_size, num_workers):
    loader = Loader(train_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=OrderOption.RANDOM,
                    pipelines={
                        'image': [RandomResizedCropRGBImageDecoder((224, 224))],
                        'label': [IntDecoder(), ToTensor(), Squeeze(), ToDevice(ch.device('cuda:0'), non_blocking=False)]
                    })

    return loader

def create_val_loader(val_dataset, batch_size, num_workers):
    loader = Loader(val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.RANDOM,
            pipelines={
                'image': [CenterCropRGBImageDecoder((243, 243))],
                'label': [IntDecoder(), ToTensor(), Squeeze(), ToDevice(ch.device('cuda:0'), non_blocking=False)]
            })
    return loader

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--beton', required=True)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--out-dir', default='/tmp')
    args = parser.parse_args()

    if args.train:
        loader = create_train_loader(args.beton, 10, 1)
    else:
        loader = create_val_loader(args.beton, 10, 1)

    ims, labs = next(iter(loader))
    for i, im in enumerate(ims):
        plt.imshow(im)
        plt.savefig(os.path.join(args.out_dir, f'chungus_{i}.png'))