from train_imagenet import IMAGENET_MEAN, IMAGENET_STD, make_trainer
from pathlib import Path
from uuid import uuid4
from tqdm import tqdm
TARGET = 0

from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

Section('viewer', 'viewer').params(
    dired=Param(str, 'path to save images', required=True)
)

ffcv_trainer = make_trainer()
import torch
ch = torch
from baseline_utils import baseline_val_loader
import numpy as np
import matplotlib.pyplot as plt

baseline_loader = baseline_val_loader()

@param('viewer.dired')
def print_samples(x, _, name, dired=None):
    x = (x.permute(1, 2, 0) * IMAGENET_STD) + IMAGENET_MEAN
    x = (x * 255).numpy().astype(np.uint8)
    parent = Path(f'{dired}/{name}').expanduser()
    plt.imsave(f'{parent}/{uuid4()}.png', x, vmin=0, vmax=255)
    plt.show()

loaders = [baseline_loader, ffcv_trainer.val_loader]
names = ['baseline', 'ffcv']

@param('viewer.dired')
def save_all(loader, name, dired=None):
    parent = Path(f'{dired}/{name}').expanduser()
    parent.mkdir(parents=True)
    for images, target in tqdm(loader):
        sel = target == TARGET
        if sel.sum() > 0:
            images, target = images[sel].cpu(), target[sel].cpu()
            for img, targ in zip(images, target):
                print_samples(img, targ, name)

with ch.no_grad():
    for loader, name in zip(loaders, names):
        print(name)
        save_all(loader, name)
