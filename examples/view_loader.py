import matplotlib
matplotlib.use('module://imgcat')

from train_imagenet import IMAGENET_MEAN, IMAGENET_STD, make_trainer
from pathlib import Path
from uuid import uuid4
from tqdm import tqdm
TARGET = 0

from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

ffcv_trainer = make_trainer()
import torch
ch = torch
import numpy as np
import matplotlib.pyplot as plt

from uuid import uuid4

dired = Path('/tmp/') / str(uuid4())
dired.mkdir()
print(dired)

def print_samples(x,name):
    x = (x.permute(1, 2, 0) * IMAGENET_STD) + IMAGENET_MEAN
    x = (x * 255).numpy().astype(np.uint8)
    plt.imsave(dired/ (name + '.png'), x, vmin=0, vmax=255)

loaders = [ffcv_trainer.val_loader, ffcv_trainer.train_loader]
names = ['val', 'train']

for loader, name in zip(loaders, names):
    for _, (inputs, labels) in enumerate(loader):
        inputs = inputs.cpu()
        for k in range(5):
            print_samples(inputs[k], name + '_' + str(k))
        break
