import torch as ch
from torch import nn
import numpy as np

def gpu_mixup(images, targets, mixup_alpha):
    lam = np.random.beta(mixup_alpha, mixup_alpha)

    images[1:] = images[1:] * lam + images[:-1] * (1 - lam) 
    new_targets = ch.cat([targets[:1], targets[:-1]], dim=0)
    return images, targets, new_targets, lam
