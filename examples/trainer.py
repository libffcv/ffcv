"""
Generic class for model training.
"""
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib as mpl
import torch.optim as optim
import json
from abc import abstractmethod
import os
from time import time
from functools import partial
from uuid import uuid4

import numpy as np
from torch._C import memory_format
import torchmetrics
from fastargs import Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf
from torch.cuda.amp import autocast
from tqdm import tqdm
from optimizations import LabelSmoothSoftmaxCEV1

import torch as ch
ch.backends.cudnn.benchmark = True
# ch.autograd.set_detect_anomaly(True)
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

mpl.use('module://imgcat')

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', required=True),
    val_dataset=Param(str, '.dat file to use for validation', required=True),
    num_workers=Param(int, 'The number of workers', required=True)
)

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'log location', required=True)
)

Section('training', 'training hyper param stuff').params(
    batch_size=Param(int, 'The batch size', default=512),
    optimizer=Param(And(str, OneOf(['sgd'])), 'The optimizer', default='sgd'),
    lr=Param(float, 'learning rate', default=0.5),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=4e-5),
    epochs=Param(int, 'number of epochs', default=24),
    lr_peak_epoch=Param(float, 'Epoch at which LR peaks', default=5.),
    label_smoothing=Param(float, 'label smoothing parameter', default=0.),
    distributed=Param(int, 'is distributed?', default=0)
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=512),
    resolution=Param(
        int, 'The size of the final resized validation image', default=224),
    lr_tta=Param(int, 'should do lr flipping/avging at test time', default=1)
)

Section('distributed').enable_if(lambda cfg: cfg['training.distributed'] == 1).params(
    world_size=Param(int, 'number gpus', default=1),
    addr=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12355')
)

class Trainer():
    def __init__(self, all_params, gpu=0):
        self.all_params = all_params
        self.gpu = gpu
        self.model, self.scaler = self.create_model_and_scaler()
        self.train_loader = self.create_train_loader()
        self.create_optimizer(len(self.train_loader))
        self.val_loader = self.create_val_loader()
        self.train_accuracy = torchmetrics.Accuracy(
            compute_on_step=False).to(self.gpu)
        self.val_meters = {
            'top_1': torchmetrics.Accuracy(compute_on_step=False).to(self.gpu),
            'top_5': torchmetrics.Accuracy(compute_on_step=False, top_k=5).to(self.gpu)
        }
        self.uid = str(uuid4())

    @abstractmethod
    def create_train_loader(self, train_dataset, batch_size, num_workers):
        raise NotImplementedError

    @abstractmethod
    def create_val_loader(self, val_dataset, batch_size, num_workers, crop_size,
                          resolution):
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
    @param('training.label_smoothing')
    def create_optimizer(self, iters_per_epoch, lr, momentum, optimizer,
                         weight_decay, epochs, lr_peak_epoch, label_smoothing):
        optimizer = optimizer.lower()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr,
                                   momentum=momentum, weight_decay=weight_decay)

        schedule = (np.arange(epochs * iters_per_epoch + 1) + 1) / \
            iters_per_epoch
        schedule = np.interp(schedule, [0, lr_peak_epoch, epochs], [0, 1, 0])
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, schedule.__getitem__)
        self.loss = ch.nn.CrossEntropyLoss()

    def train_loop(self):
        model = self.model
        model.train()
        losses = []

        log_iters = [0, 1, len(self.train_loader) - 1]
        iterator = tqdm(self.train_loader)

        for ix, (images, target) in enumerate(iterator):
            images = images.to(memory_format=ch.channels_last,
                               non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast():
                output = self.model(images)
                loss_train = self.loss(output, target)
                losses.append(loss_train.detach())
                self.train_accuracy(output, target)

            this_bs = images.shape[0]
            lr_modifier = self.scheduler.get_lr()
            group_lrs = []
            for group_ix, group in enumerate(self.optimizer.param_groups):
                group_lrs.append(group['lr'])

            this_resolution = images.shape[2:]
            msg = f'bs={this_bs},lr_ratio={lr_modifier},group_lrs={group_lrs},res={this_resolution}'
            iterator.set_description(msg)

            if ix in log_iters:
                print(f'it={ix}', msg)

            self.scaler.scale(loss_train).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

        accuracy = self.train_accuracy.compute().item()
        self.train_accuracy.reset()
        loss = ch.stack(losses).mean().item()
        return loss, accuracy

    @param('validation.lr_tta')
    def val_loop(self, lr_tta):
        model = self.model
        model.eval()
        losses = []

        with ch.no_grad():
            for images, target in tqdm(self.val_loader):
                images = images.to(memory_format=ch.channels_last,
                                   non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)

                with autocast():
                    output = self.model(images)
                    if lr_tta:
                        output += self.model(ch.flip(images, dims=[3]))

                    loss_val = self.loss(output, target)
                    losses.append(loss_val.detach())
                    [meter(output, target)
                     for meter in self.val_meters.values()]

        stats = {
            k: meter.compute().item() for k, meter in self.val_meters.items()
        }

        [meter.reset() for meter in self.val_meters.values()]
        loss = ch.stack(losses).mean().item()
        return loss, stats

    @param('logging.folder')
    def initialize_logger(self, folder):
        folder = Path(folder).absolute()

        self.logging_fp = str(folder / f'{self.uid}.log')
        self.logging_fd = open(self.logging_fp, 'w+')
        self.params_fp = str(folder / f'{self.uid}-params.json')

        self.start_time = time()
        hyper_params = {
            '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()}
        with open(self.params_fp, 'w+') as handle:
            json.dump(hyper_params, handle)

    def log(self, content):
        cur_time = time()
        self.logging_fd.write(json.dumps({
            'timestamp': cur_time,
            'relative_time': cur_time - self.start_time,
            **content
        }))
        self.logging_fd.write('\n')
        self.logging_fd.flush()
        print(f'>>> Logging file: {self.logging_fp}')
        print(f'>>> Params file: {self.params_fp}')

    @param('training.epochs')
    def train(self, epochs):
        print('Started training...')
        self.initialize_logger()
        for epoch in range(epochs):
            train_loss, train_acc = self.train_loop()
            self.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'epoch': epoch,
            })
        val_loss, val_stats = self.val_loop()
        self.log({
            'val_loss': val_loss,
            **val_stats
        })

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


Section('distributed').enable_if(lambda cfg: cfg['training.distributed'] == 1).params(
    world_size=Param(int, 'number gpus', default=1),
    addr=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12355')
)

class DistributedTrainer(Trainer):
    @param('distributed.world_size')
    def launch_distributed(config, world_size):
        def train(rank):
            trainer = DistributedTrainer(config, rank)
            trainer.train()
            trainer.cleanup()

        mp.spawn(train, args=tuple(), nprocs=world_size, join=True)

    @param('distributed.world_size')
    @param('distributed.addr')
    @param('distributed.port')
    def __init__(self, *args, gpu, world_size, addr, port, **kwargs):
        super().__init__(*args, **kwargs)
        assert type(gpu) is int
        assert type(addr) is str
        assert type(port) is str
        assert type(world_size) is int

        os.environ['MASTER_ADDR'] = addr
        os.environ['MASTER_PORT'] = port
        dist.init_process_group('nccl', rank=gpu, world_size=world_size)
        # self.addr = addr
        # self.port = port

    def cleanup(self):
        dist.destroy_process_group()

    # put ALL distributed stuff in here (ie all device checks + checks of whether
    #     it is distributed etc)
    def log(self, *args, **kwargs):
        if self.gpu == 0:
            return super().log(*args, **kwargs)

    def initialize_logger(self, *args, **kwargs):
        if self.gpu == 0:
            return super().initialize_logger(*args, **kwargs)

