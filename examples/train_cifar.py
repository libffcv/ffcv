import torch as ch
from torch.cuda.amp import GradScaler
import subprocess
from ffcv.pipeline.compiler import Compiler
from ffcv.transforms.ops import ToTorchImage
from trainer import Trainer
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import Cutout, RandomHorizontalFlip, ToTensor, Collate, ToDevice, Squeeze, Convert
from ffcv.fields.rgb_image import SimpleRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from torchvision.transforms import Normalize
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf
from argparse import ArgumentParser
from cifar_models import models, AffineAugmentation
import numpy as np

Section('model', 'model details').params(
    arch=Param(And(str, OneOf(models.keys())), 'the architecture to use', required=True)
)

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR_STD = np.array([0.2023, 0.1994, 0.2010])

class CIFARTrainer(Trainer):
    @param('data.train_dataset')
    @param('training.batch_size')
    @param('data.num_workers')
    def create_train_loader(self, train_dataset, batch_size, num_workers):
        loader = Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.RANDOM,
                        pipelines={
                            'image': [
                                SimpleRGBImageDecoder(), 
                                ToTensor(), 
                                ToDevice(ch.device('cuda:0')), 
                                ToTorchImage(), 
                                Convert(ch.float16),
                                Normalize((CIFAR_MEAN * 255).tolist(), (CIFAR_STD * 255).tolist()),
                                AffineAugmentation(4)
                            ],
                            'label': [IntDecoder(), ToTensor(), Squeeze(), ToDevice(ch.device('cuda:0'))]
                        })

        return loader

    @param('data.val_dataset')
    @param('validation.batch_size')
    @param('data.num_workers')
    @param('validation.crop_size')
    @param('validation.resolution')
    # TODO: remove crop_size and resolution arguments for CIFAR unrolling
    def create_val_loader(self, val_dataset, batch_size, num_workers, crop_size, resolution):
        loader = Loader(val_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                order=OrderOption.RANDOM,
                pipelines={
                    'image': [
                        SimpleRGBImageDecoder(), 
                        ToTensor(), 
                        ToDevice(ch.device('cuda:0')), 
                        ToTorchImage(), 
                        Convert(ch.float16),
                        Normalize((CIFAR_MEAN * 255).tolist(), (CIFAR_STD * 255).tolist())
                    ],
                    'label': [IntDecoder(), ToTensor(), Squeeze(),
                              ToDevice(ch.device('cuda:0'))]
                })
        return loader

    @param('model.arch')
    def create_model_and_scaler(self, arch):
        scaler = GradScaler()
        model = models[arch]()
        model = model.to(memory_format=ch.channels_last)
        model.cuda(self.gpu)
        return model, scaler

# OVERRIDES = {
#     'validation.crop_size':None,
#     'validation.resolution':None
# }

if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    trainer = CIFARTrainer(config)
    trainer.train()
    trainer.log_val()