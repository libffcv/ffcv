from multiprocessing import Value
import torch as ch
from pathlib import Path
from torch.cuda.amp import GradScaler
import numpy
import matplotlib
matplotlib.use('module://itermplot')
import matplotlib.pyplot as plt

from ffcv.pipeline.compiler import Compiler
import antialiased_cnns
import time
from uuid import uuid4
from ffcv.transforms.ops import ToTorchImage
from trainer import Trainer
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import Cutout, RandomHorizontalFlip, ToTensor, Collate, ToDevice, Squeeze, Convert
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from torchvision.transforms import Normalize, ColorJitter
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf
from argparse import ArgumentParser
import numpy as np
from torchvision import models
import torch.optim as optim


Section('model', 'model details').params(
    arch=Param(And(str, OneOf(models.__dir__())),
               'the architecture to use', required=True),
    antialias=Param(int, 'use blurpool or not', default=0)
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=224),
    max_res=Param(int, 'the maximum (starting) resolution', default=224),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=0)
)

Section('training', 'training hyper param stuff').params(
    step_ratio=Param(float, 'learning rate step ratio', default=0.1),
    step_length=Param(int, 'learning rate step length', default=30),
    lr_schedule_type=Param(OneOf(['step', 'cyclic']), 'step or cyclic schedule?',
                           required=True),
    eval_only=Param(int, 'eval only?', default=0),
    pretrained=Param(int, 'is pretrained?', default=0)
)

Section('distributed').params(
    world_size=Param(int, 'number gpus', default=1),
    addr=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12355')
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
DEFAULT_CROP_RATIO = 224/256

class ImageNetTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_schedules()

    @param('training.epochs')
    @param('training.lr_schedule_type')
    @param('training.lr_peak_epoch')
    @param('training.step_ratio')
    @param('training.step_length')
    @param('resolution.min_res')
    @param('resolution.max_res')
    @param('resolution.end_ramp')
    @param('training.batch_size')
    def setup_schedules(self, epochs, lr_schedule_type, lr_peak_epoch, step_ratio,
                        step_length, min_res, max_res, end_ramp, batch_size):
        assert min_res <= max_res
        assert step_ratio <= 1

        def schedule(epoch):
            if end_ramp == 0:
                result = max_res
            else:
                diff = max_res - min_res
                result = min_res
                result += min(1, epoch / end_ramp) * diff
                result = int(np.round(result / 32) * 32)
            return (result, result)

        res_sched = [schedule(ep) for ep in range(epochs)]

        # per epoch schedules
        self.resolution_schedule = res_sched

        def bs_for_res(res):  # TODO: n^2 or n scaling?
            frac = max_res // res
            return frac * batch_size

        self.batch_size_schedule = [bs_for_res(rs[0]) for rs in res_sched]

        # We now have to make per iteration learning rate schedules, we scale
        # the learning rate linearly with the batch size
        # Schedules: should be in [0, 1]; they are relatively scaled to the
        #   overall learning rate set in args.lr
        if lr_schedule_type == 'cyclic':
            # schedule: from 1 to `epochs` to avoid 0 epoch at 0
            schedule_xs = np.arange(epochs + 1) + 1  # { 1, 2, ..., epochs }
            # now interpolate schedule along the ``mountain'' path
            xs, ys = [0, lr_peak_epoch, epochs + 1], [0, 1, 0]
            schedule = np.interp(schedule_xs, xs, ys)
        elif lr_schedule_type == 'step':
            schedule = []
            for this_epoch in range(epochs + 1):
                plateau_number = this_epoch // step_length
                lr_modifier = step_ratio**plateau_number
                schedule.append(lr_modifier)

            schedule = np.array(schedule)

        # we now simulate training to yield per iteration learning rates
        # we train only from the first nonzero epoch lr to the last nonzero
        # epoch lr, interpolating inbetween
        # dataset_size = len(self.train_loader) * self.train_loader.batch_size
        try:
            dataset_size = self.train_loader.reader.num_samples
        except:
            dataset_size = len(self.train_loader.dataset)

        learning_rates = []
        epochs = []
        for start_epoch, this_bs in enumerate(self.batch_size_schedule):
            lr_scaling = this_bs / batch_size  # keep effective lr/bs constant
            start_epoch_lr = schedule[start_epoch] * lr_scaling
            end_epoch_lr = schedule[start_epoch + 1] * lr_scaling
            num_iterations = dataset_size // batch_size  # we always drop last

            xs, ys = [0, num_iterations], [start_epoch_lr, end_epoch_lr]
            schedule_xs = np.arange(num_iterations + 1)
            this_epoch_lrs = np.interp(schedule_xs, xs, ys)

            # grab up until the final one
            learning_rates.append(this_epoch_lrs[:-1])
            epochs.append(start_epoch + schedule_xs[:-1]/num_iterations)

        learning_rates = np.concatenate(learning_rates, axis=0)
        epochs = np.concatenate(epochs, axis=0)

        _, ax = plt.subplots(figsize=(6, 2))
        ax.plot(epochs, learning_rates, label='learning rate')
        ax.legend()
        plt.show()

        _, ax = plt.subplots(figsize=(6, 2))
        ax.plot(np.arange(len(self.batch_size_schedule)),
            self.batch_size_schedule, label='bs')
        ax.legend()
        plt.show()

        _, ax = plt.subplots(figsize=(6, 2))
        ax.plot(np.arange(len(self.batch_size_schedule)),
            [x[0] for x in res_sched], label='resolution')
        ax.legend()
        plt.show()

        learning_rates = np.concatenate([learning_rates, np.zeros((1,))],
                                        axis=0)

        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, learning_rates.__getitem__
        )

    @param('training.lr')
    @param('training.momentum')
    @param('training.optimizer')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    def create_optimizer(self, _, lr, momentum, optimizer, weight_decay,
                         label_smoothing):
        assert optimizer == 'sgd'

        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=lr,
                                   momentum=momentum,
                                   weight_decay=weight_decay)

        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    @param('data.train_dataset')
    @param('training.batch_size')
    @param('data.num_workers')
    def create_train_loader(self, train_dataset, batch_size, num_workers):
        train_path = Path(train_dataset)
        assert train_path.is_file()
        self.decoder = RandomResizedCropRGBImageDecoder((224, 224))
        loader = Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.RANDOM,
                        os_cache=True,
                        drop_last=True,
                        pipelines={
                            'image': [
                                self.decoder,
                                ToTensor(),
                                ToDevice(ch.device('cuda:0'),
                                        non_blocking=False),
                                ToTorchImage(),
                                Convert(ch.float16),
                                Normalize((IMAGENET_MEAN * 255).tolist(),
                                        (IMAGENET_STD * 255).tolist()),
                            ],
                            'label': [IntDecoder(), ToTensor(), Squeeze(),
                                    ToDevice(ch.device('cuda:0'),
                                    non_blocking=False)]
                        })

        return loader

    @param('data.val_dataset')
    @param('validation.batch_size')
    @param('data.num_workers')
    @param('validation.resolution')
    def create_val_loader(self, val_dataset, batch_size, num_workers,
                          resolution, ratio=DEFAULT_CROP_RATIO):
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=ratio)
        loader = Loader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': [
                                cropper,
                                ToTensor(),
                                ToDevice(ch.device('cuda:0'),
                                        non_blocking=False),
                                ToTorchImage(),
                                Convert(ch.float16),
                                Normalize((IMAGENET_MEAN * 255).tolist(),
                                        (IMAGENET_STD * 255).tolist())
                            ],
                            'label': [IntDecoder(), ToTensor(), Squeeze(),
                                    ToDevice(ch.device('cuda:0'),
                                    non_blocking=False)]
                        })
        return loader

    @param('training.epochs')
    def train(self, epochs):
        # assert self.train_loader.drop_last, 'drop last must be enabled!'
        for epoch in range(epochs):
            train_loss, train_acc = self.train_loop(epoch)

            extra_dict = {
                'train_loss': train_loss,
                'epoch': epoch,
                'train_acc': train_acc,
            }

            self.eval_and_log(extra_dict)

    def eval_and_log(self, extra_dict={}):
        start_val = time.time()
        _, stats = self.val_loop()
        val_time = time.time() - start_val

        self.log(dict({
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'top_1': stats['top_1'],
            'top_5': stats['top_5'],
            'val_time': val_time
        }, **extra_dict))

    @param('model.arch')
    @param('model.antialias')
    @param('training.pretrained')
    # @param('distributed.world_size')
    def create_model_and_scaler(self, arch, antialias, pretrained):
        scaler = GradScaler()
        if not antialias:
            model = getattr(models, arch)(pretrained=pretrained)
        else:
            raise ValueError('dont do this eom')
            # model = getattr(antialiased_cnns, arch)(pretrained=False)

        model = model.to(memory_format=ch.channels_last)
        model.cuda(self.gpu)
        return model, scaler

def make_trainer():
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    trainer = ImageNetTrainer(config)
    return trainer

@param('training.eval_only')
def execute(trainer, eval_only=False):
    print(eval_only)
    if not eval_only:
        trainer.train()
    else:
        trainer.eval_and_log()
    return trainer

def main():
    trainer = make_trainer()
    return execute(trainer)

if __name__ == "__main__":
    main()