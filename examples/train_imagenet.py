import torch as ch
from torch.cuda.amp import GradScaler
from ffcv.pipeline.compiler import Compiler
# import antialiased_cnns
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
    arch=Param(And(str, OneOf(models.__dir__())), 'the architecture to use', required=True),
    antialias=Param(bool, 'use blurpool or not', is_flag=True)
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=224),
    max_res=Param(int, 'the maximum (starting) resolution', default=224),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=0)
)

IMAGENET_MEAN = np.array([0.4914, 0.4822, 0.4465])
IMAGENET_STD = np.array([0.2023, 0.1994, 0.2010])

class ImageNetTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_sched()

    @param('training.epochs')
    @param('resolution.min_res')
    @param('resolution.max_res')
    @param('resolution.end_ramp')
    def setup_sched(self, epochs, min_res, max_res, end_ramp):
        def schedule(epoch):
            diff = max_res - min_res
            result = min_res
            result += min(1, epoch / end_ramp) * diff
            result = int(np.round(result / 32) * 32)
            return (result, result)
        self.resolution_schedule = [schedule(ep) for ep in range(epochs)]
        # self.batch_size_schedule = [256 * 160 // rs[0] for rs in self.resolution_schedule]
        self.batch_size_schedule = [1024 for rs in self.resolution_schedule]

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
                                ToDevice(ch.device('cuda:0'), non_blocking=False), 
                                ToTorchImage(), 
                                Convert(ch.float16),
                                Normalize((IMAGENET_MEAN * 255).tolist(), (IMAGENET_STD * 255).tolist())
                            ],
                            'label': [IntDecoder(), ToTensor(), Squeeze(), ToDevice(ch.device('cuda:0'), non_blocking=False)]
                        })

        return loader

    @param('data.val_dataset')
    @param('validation.batch_size')
    @param('data.num_workers')
    @param('validation.crop_size')
    @param('validation.resolution')
    def create_val_loader(self, val_dataset, batch_size, num_workers, crop_size,
                          resolution):
        loader = Loader(val_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                order=OrderOption.RANDOM,
                pipelines={
                    'image': [
                        RandomResizedCropRGBImageDecoder((224, 224)), 
                        ToTensor(), 
                        ToDevice(ch.device('cuda:0'), non_blocking=False), 
                        ToTorchImage(), 
                        Convert(ch.float16),
                        Normalize((IMAGENET_MEAN * 255).tolist(), (IMAGENET_MEAN * 255).tolist())
                    ],
                    'label': [IntDecoder(), ToTensor(), Squeeze(), ToDevice(ch.device('cuda:0'), non_blocking=False)]
                })
        return loader

    @param('training.epochs')
    def train(self, epochs):
        for epoch in range(epochs):
            self.decoder.output_size = self.resolution_schedule[epoch]
            self.train_loader.batch_size = self.batch_size_schedule[epoch]
            train_loss, train_acc = self.train_loop()
            self.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'epoch': epoch,
            })
        self.val_loop()

    @param('model.arch')
    @param('model.antialias')
    def create_model_and_scaler(self, arch, antialias):
        scaler = GradScaler()
        if not antialias:
            model = getattr(models, arch)()
        else:
            model = getattr(antialiased_cnns, arch)(pretrained=False)

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
    trainer = ImageNetTrainer(config)
    trainer.train()
