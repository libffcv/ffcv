import torch as ch
from torch.cuda.amp import GradScaler
from ffcv.pipeline.compiler import Compiler
from ffcv.transforms.ops import ToTorchImage
from trainer import Trainer
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import Cutout, RandomHorizontalFlip, ToTensor, Collate, ToDevice, Squeeze, Convert
from ffcv.fields.rgb_image import RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from torchvision.transforms import Normalize
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf
from argparse import ArgumentParser
import numpy as np
from torchvision import models

Section('model', 'model details').params(
    arch=Param(And(str, OneOf(models.__dir__())), 'the architecture to use', required=True)
)

IMAGENET_MEAN = np.array([0.4914, 0.4822, 0.4465])
IMAGENET_STD = np.array([0.2023, 0.1994, 0.2010])

class ImageNetTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolution_schedule = []

    @param('data.train_dataset')
    @param('training.batch_size')
    @param('data.num_workers')
    def create_train_loader(self, train_dataset, batch_size, num_workers):
        self.decoder = RandomResizedCropRGBImageDecoder((224, 224))
        loader = Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.RANDOM,
                        pipelines={
                            'image': [
                                self.decoder,
                                ToTensor(), 
                                ToDevice(ch.device('cuda:0')), 
                                ToTorchImage(), 
                                Convert(ch.float16),
                                Normalize((IMAGENET_MEAN * 255).tolist(), (IMAGENET_STD * 255).tolist())
                            ],
                            'label': [IntDecoder(), ToTensor(), Squeeze(), ToDevice(ch.device('cuda:0'))]
                        })

        return loader

    @param('data.val_dataset')
    @param('validation.batch_size')
    @param('data.num_workers')
    @param('validation.crop_size')
    @param('validation.resolution')
    def create_val_loader(self, val_dataset, batch_size, num_workers, crop_size, resolution):
        loader = Loader(val_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                order=OrderOption.RANDOM,
                pipelines={
                    'image': [
                        RandomResizedCropRGBImageDecoder((224, 224)), 
                        ToTensor(), 
                        ToDevice(ch.device('cuda:0')), 
                        ToTorchImage(), 
                        Convert(ch.float16),
                        Normalize((CIFAR_MEAN * 255).tolist(), (CIFAR_STD * 255).tolist())
                    ],
                    'label': [IntDecoder(), ToTensor(), Squeeze(), ToDevice(ch.device('cuda:0'))]
                })
        return loader

    @param('training.epochs')
    def train(self, epochs):
        for epoch in range(epochs):
            self.decoder.update_resolution(self.resolution_schedule[epoch])
            train_loss, train_acc = self.train_loop()
            self.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'epoch': epoch,
            })

    @param('model.arch')
    def create_model_and_scaler(self, arch):
        scaler = GradScaler()
        model = getattr(models, arch)()
        model = model.to(memory_format=ch.channels_last)
        model.cuda(self.gpu)
        return model, scaler

if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    trainer = CIFARTrainer(config)
    trainer.train()

"""
Section('training.resolution_schedule',
        'How the resolution increases during training').params(
            min_resolution=Param(int, 'resolution at the first epoch', default=160),
            end_ramp=Param(int, 'At which epoch should we end increasing the resolution',
                           default=20),
            max_resolution=Param(int, 'Resolution we reach at end', default=160),
        )

Section('optimizations').params(
    label_smoothing=Param(float, 'alpha for label smoothing'),
    blurpool=Param(int, 'Whether to use blurpool layers', default=1),
    tta=Param(int, 'Whether to use test-time augmentation', default=1)
)

class TTAModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if self.training: return self.model(x)
        return self.model(x) + self.model(ch.flip(x, dims=[3]))

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
"""