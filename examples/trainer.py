"""
Generic class for model training.
"""
import json
from abc import abstractmethod
from os import path
from time import time
from uuid import uuid4

import numpy as np
from torch._C import memory_format
import torchmetrics
from fastargs import Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf
from torch.cuda.amp import autocast
from tqdm import tqdm

import torch as ch
# ch.backends.cudnn.benchmark = True
# ch.autograd.set_detect_anomaly(True)
# ch.autograd.profiler.emit_nvtx(False)
# ch.autograd.profiler.profile(False)

import torch.nn.functional as F
import torch.optim as optim

import matplotlib as mpl
mpl.use('module://imgcat')
from matplotlib import pyplot as plt

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', required=True),
    val_dataset=Param(str, '.dat file to use for validation', required=True),
    num_workers=Param(int, 'The number of workers', default=16),
    gpu=Param(int, 'Which GPU to use', default=0)
)

Section('logging', 'how to log stuff').params(folder=Param(str, 'log location', default='/tmp'))

Section('training', 'training hyper param stuff').params(
    batch_size=Param(int, 'The batch size', default=512),
    optimizer=Param(And(str, OneOf(['sgd'])), 'The optimizer', default='sgd'),
    lr=Param(float, 'learning rate', default=0.5),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=5e-4),
    epochs=Param(int, 'number of epochs', default=24),
    lr_peak_epoch=Param(float, 'Epoch at which LR peaks', default=5.),
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=512),
    crop_size=Param(int, 'The size of the crop before resizing to resolution', default=243),
    resolution=Param(int, 'The size of the final resized validation image', default=224)
)

class Trainer():
    @param('data.gpu')
    def __init__(self, all_params, gpu):
        self.all_params = all_params
        self.gpu = gpu
        self.model, self.scaler = self.create_model_and_scaler()
        self.train_loader = self.create_train_loader()
        self.create_optimizer(len(self.train_loader))
        self.val_loader = self.create_val_loader()
        self.train_accuracy = torchmetrics.Accuracy(compute_on_step=False).cuda(self.gpu)
        self.val_meters = {
            'top_1': torchmetrics.Accuracy(compute_on_step=False).cuda(self.gpu),
            'top_5': torchmetrics.Accuracy(compute_on_step=False, top_k=5).cuda(self.gpu)
        }
        self.uid = str(uuid4())
        self.initialize_logger()
    
    @abstractmethod
    def create_train_loader(self, train_dataset, batch_size, num_workers):
        raise NotImplementedError
    
    @abstractmethod
    def create_val_loader(self, val_dataset, batch_size, num_workers, crop_size, resolution):
        raise NotImplementedError
    
    @abstractmethod
    def create_model(self, architecture, tta):
        raise NotImplementedError

    @param('training.lr')
    @param('training.optimizer')
    @param('training.momentum')
    @param('training.weight_decay')
    @param('training.epochs')
    @param('training.lr_peak_epoch')
    def create_optimizer(self, iters_per_epoch, lr, momentum, optimizer,
                         weight_decay, epochs, lr_peak_epoch):
        optimizer = optimizer.lower()
        self.optimizer = optim.SGD(self.model.parameters(),
                                     lr=lr,
                                     momentum=momentum,
                                     weight_decay=weight_decay)

        print(iters_per_epoch)
        # schedule = (np.arange(epochs * iters_per_epoch + 1) + 1) / iters_per_epoch
        schedule = (np.arange(epochs + 1) + 1) 
        # add 1 to avoid 0 learning rate at the end.
        schedule = np.interp(schedule, [0, lr_peak_epoch, epochs + 1], [0, 1, 0])
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, schedule.__getitem__)

    def train_loop(self):
        model = self.model
        model.train()
        losses = []
        for images, target in tqdm(self.train_loader):
            images = images.to(memory_format=ch.channels_last)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast():
                output = self.model(images)
                loss_train = F.cross_entropy(output, target)
                losses.append(loss_train.detach())
                self.train_accuracy(output, target)

            self.scaler.scale(loss_train).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        self.scheduler.step()

        accuracy = self.train_accuracy.compute().item()
        self.train_accuracy.reset()
        loss = ch.stack(losses).mean().item()
        return loss, accuracy

    def val_loop(self):
        model = self.model
        model.eval()
        losses = []

        with ch.inference_mode():
            for images, target in tqdm(self.val_loader):
                images = images.to(memory_format=ch.channels_last)
                self.optimizer.zero_grad(set_to_none=True)

                with autocast():
                    output = self.model(images)
                    loss_val = F.cross_entropy(output, target)
                    losses.append(loss_val.detach())
                    [meter(output, target) for meter in self.val_meters.values()]

        stats = {k: meter.compute().item() for k, meter in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        loss = ch.stack(losses).mean().item()
        print(stats)
        return loss, stats

    @param('logging.folder')
    def initialize_logger(self, folder):
        self.logging_fp = open(path.join(folder, f'{self.uid}.log'), 'w+')
        print(path.join(folder, f'{self.uid}.log'))
        self.start_time = time()
        hyper_params = {'.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()}
        with open(path.join(folder, f'{self.uid}-params.json'), 'w+') as handle:
            json.dump(hyper_params, handle)

    def log(self, content):
        cur_time = time()
        self.logging_fp.write(json.dumps({
            'timestamp': cur_time,
            'relative_time': cur_time - self.start_time,
            **content
        }))
        self.logging_fp.write('\n')
        self.logging_fp.flush()

    @param('training.epochs')
    def train(self, epochs):
        for epoch in range(epochs):
            train_loss, train_acc = self.train_loop()
            self.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                # 'val_loss': val_loss,
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                # **val_stats
            })
        val_loss, val_stats = self.val_loop()
