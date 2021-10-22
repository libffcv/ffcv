from tqdm import tqdm
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import Cutout, RandomHorizontalFlip, ToTensor, Collate

import numpy as np
from uuid import uuid4
from time import time
import json
from torch.cuda.amp import GradScaler, autocast

from os import path
from argparse import ArgumentParser, Namespace

import torch as ch
ch.backends.cudnn.benchmark = True
ch.autograd.set_detect_anomaly(False)
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

import torchvision
import torchmetrics

from fastargs import Section, Param
from fastargs.validation import And, OneOf
from fastargs.decorators import param, section
from fastargs import get_current_config

MODEL_NAMES = sorted(
    name
    for name in torchvision.models.__dict__
    if name.islower() and not name.startswith("__") and callable(torchvision.models.__dict__[name])
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', required=True),
    val_dataset=Param(str, '.dat file to use for validation', required=True),
    num_workers=Param(int, 'The number of workers', default=16),
    gpu=Param(int, 'Which GPU to use', default=0)
)

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'Where to put the logs', default='/tmp')
)

Section('training', 'training hyper param stuff').params(
    architecture=Param(And(str, OneOf(MODEL_NAMES)),
                       'The architecture to use', default='resnet50'),
    batch_size=Param(int, 'The batch size', default=512),
    optimizer=Param(And(str, OneOf(['sgd', 'lamb', 'sam'])), 'The optimizer', default='sgd'),
    lr=Param(float, 'learning rate', default=0.1),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'learning rate', default=1e-4),
    epochs=Param(int, 'The number of epochs', default=30),
    lr_peak_epoch=Param(float, 'Epoch at which LR peaks', default=1.),
)

Section('optimizations').params(
    label_smoothing=Param(float, 'alpha for label smoothing'),
    blurpool=Param(int, 'Whether to use blurpool layers', default=1),
    tta=Param(int, 'Whether to use test-time augmentation', default=1)
)

Section('training.resolution_schedule',
        'How the resolution increases during training').params(
            min_resolution=Param(int, 'resolution at the first epoch', default=160),
            end_ramp=Param(int, 'At which epoch should we end increasing the resolution',
                           default=20),
            max_resolution=Param(int, 'Resolution we reach at end', default=160),
        )

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=512),
    crop_size=Param(int, 'The size of the crop before resizing to resolution', default=243),
    resolution=Param(int, 'The size of the final resized validation image', default=224)
)

@section('training.resolution_schedule')
@param('min_resolution')
@param('max_resolution')
@param('end_ramp')
def get_resolution_schedule(min_resolution, max_resolution, end_ramp):
    def schedule(epoch):
        diff = max_resolution - min_resolution
        result =  min_resolution
        result +=  min(1, epoch / end_ramp) * diff
        result = int(np.round(result / 32) * 32)
        return result
    return schedule

@param('data.train_dataset')
@param('training.batch_size')
@param('data.num_workers')
def get_train_dataset(train_dataset, batch_size, num_workers):
    loader = Loader(train_dataset,
                    batch_size=batch_size,
                    num_workers=16,
                    order=OrderOption.RANDOM)
    loader.pipelines['image'] = [
        Cutout(8),
        RandomHorizontalFlip(0.5),
        Collate(),
        ToTensor()
    ]
    return loader

@param('data.val_dataset')
@param('validation.batch_size')
@param('data.num_workers')
@param('validation.crop_size')
@param('validation.resolution')
def get_val_dataset(val_dataset, batch_size, num_workers, crop_size, resolution):
    loader = Loader(val_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=OrderOption.RANDOM)
    return loader

class TTAModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if self.training: return self.model(x)
        return self.model(x) + self.model(ch.flip(x, dims=[3]))

class Trainer():
    @param('data.gpu')
    def __init__(self, all_params, gpu):
        self.all_params = all_params
        self.gpu = gpu
        self.create_model()
        self.train_loader = get_train_dataset()
        self.create_optimizer(len(self.train_loader))
        self.val_loader = get_val_dataset()
        # self.normalization = transforms.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                #   std=[0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda(gpu)
        self.train_accuracy = torchmetrics.Accuracy(compute_on_step=False).cuda(self.gpu)
        self.val_accuracy = torchmetrics.Accuracy(compute_on_step=False).cuda(self.gpu)
        self.val_top5 = torchmetrics.Accuracy(compute_on_step=False, top_k=5).cuda(self.gpu)
        self.uid = str(uuid4())
        self.initialize_logger()

    @param('training.architecture')
    @param('optimizations.tta')
    def create_model(self, architecture, tta):
        self.scaler = GradScaler()
        model = torchvision.models.__dict__[architecture]()
        if tta: model = TTAModel(model)
        model = model.to(memory_format=ch.channels_last)
        model.cuda(self.gpu)
        self.model = model

    @param('training.lr')
    @param('training.optimizer')
    @param('training.momentum')
    @param('training.weight_decay')
    @param('training.epochs')
    @param('training.lr_peak_epoch')
    def create_optimizer(self, iters_per_epoch, lr, momentum, optimizer,
                         weight_decay, epochs, lr_peak_epoch):
        optimizer = optimizer.lower()
        assert optimizer in ['lamb', 'sgd']
        self.optimizer = optim.SGD(self.model.parameters(),
                                     lr=lr,
                                     momentum=momentum,
                                     weight_decay=weight_decay)
        
        schedule = (np.arange(epochs * iters_per_epoch + 1) + 1) / iters_per_epoch
        schedule = np.interp(schedule, [0, lr_peak_epoch, epochs], [0, 1, 0])
        def lambda_schedule(t):
            return schedule[t]
            # return np.interp(float(t + 1) / iters_per_epoch, [0, lr_peak_epoch, epochs], [0, 1, 0])
        
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer,
                                               lambda_schedule)
        # self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, epochs * iters_per_epoch)

    def train_loop(self):
        model = self.model
        model.train()
        losses = []
        for iii, (images, target) in enumerate(tqdm(self.train_loader)):
            # TODO: replace with collation in the pipeline
            images = images.permute([0, 3, 1, 2])
            target = target.squeeze()
            # TODO: will gpu stuff still be here?
            images = images.cuda(self.gpu, non_blocking=True)
            target = target.cuda(self.gpu, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            images = images.float()
            # images = self.normalization(images)

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
                # TODO: replace with collation in the pipeline
                images = images.permute([0, 3, 1, 2])
                target = target.squeeze()
                # TODO: will gpu stuff still be here?
                images = images.cuda(self.gpu, non_blocking=True)
                target = target.cuda(self.gpu, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)

                images = images.float()
                # images = self.normalization(images)

                with autocast():
                    output = self.model(images)
                    loss_val = F.cross_entropy(output, target)
                    losses.append(loss_val.detach())
                    self.val_accuracy(output, target)
                    self.val_top5(output, target)

        accuracy = self.val_accuracy.compute().item()
        self.val_accuracy.reset()
        top_5 = self.val_top5.compute().item()
        self.val_top5.reset()
        loss = ch.stack(losses).mean().item()
        return loss, accuracy, top_5

    @param('logging.folder')
    def initialize_logger(self, folder):
        self.logging_fp = open(path.join(folder, f'{self.uid}.log'), 'w+')
        print(path.join(folder, f'{self.uid}.log'))
        self.start_time = time()
        with open(path.join(folder, f'{self.uid}-params.json'), 'w+') as handle:
            json.dump(self.all_params, handle)

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
            val_loss, val_acc, val_top5 = self.val_loop()
            self.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_top5': val_top5,
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'current_resolution': get_resolution_schedule()(epoch),
                'epoch': epoch
            })

@param('training.epochs')
def main(args: Namespace, epochs) -> None:
    resolution_schedule = get_resolution_schedule()
    print("Resolution schedule summary")
    print([resolution_schedule(x) for x in range(epochs)])
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    args = config.get()
    hyper_params = {'.'.join(k): config[k] for k in config.entries.keys()}
    main(hyper_params)