import torch as ch
from torch.cuda.amp import GradScaler
from ffcv.transforms.ops import ToTorchImage
from trainer import Trainer
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import Cutout, RandomHorizontalFlip, ToTensor, Collate, ToDevice, Squeeze, Convert
from torchvision.transforms import Normalize
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf
from argparse import ArgumentParser
from cifar_models import models, AffineAugmentation

Section('model', 'model details').params(
    arch=Param(And(str, OneOf(models.keys())), 'the architecture to use', required=True)
)

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

class CIFARTrainer(Trainer):
    @param('data.train_dataset')
    @param('training.batch_size')
    @param('data.num_workers')
    def create_train_loader(self, train_dataset, batch_size, num_workers):
        loader = Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.RANDOM)
        loader.pipelines['image'] = [
            Cutout(8, (int(CIFAR_MEAN[0] * 255), int(CIFAR_MEAN[1] * 255), int(CIFAR_MEAN[2] * 255))),
            # RandomHorizontalFlip(0.5),
            Collate(),
            ToTensor(),
            ToTorchImage(channels_last=False),
            Convert(ch.float32),
            Normalize(mean=[0., 0., 0.],
                      std=[255., 255., 255.],
                      inplace=False),
            Convert(ch.float16),
            ToDevice(ch.device('cuda:0')),
            AffineAugmentation(4),
            # Cutout(8, 8, 1),
            Normalize(mean=[x for x in CIFAR_MEAN],
                      std=[x for x in CIFAR_STD],
                      inplace=False).to('cuda:0'),
            # Normalize(mean=[0., 0., 0.],
                    #   std=[255., 255., 255.],
                    #   inplace=True).to('cuda:0')
        ]
        loader.pipelines['label'] = [
            Collate(),
            ToTensor(),
            Squeeze(-1),
            ToDevice('cuda:0')
        ]
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
                        order=OrderOption.RANDOM)
        loader.pipelines['image'] = [
            Collate(),
            ToTensor(),
            ToTorchImage(channels_last=False),
            Convert(ch.float32),
            Normalize(mean=[0., 0., 0.],
                      std=[255., 255., 255.],
                      inplace=False),
            Convert(ch.float16),
            ToDevice('cuda:0'),
            Normalize(mean=[x for x in CIFAR_MEAN],
                      std=[x for x in CIFAR_STD],
                      inplace=False).to('cuda:0'),
        ]
        loader.pipelines['label'] = [
            Collate(),
            ToTensor(),
            Squeeze(-1),
            ToDevice(ch.device('cuda:0'))
        ]
        return loader

    @param('model.arch')
    def create_model_and_scaler(self, arch):
        scaler = GradScaler()
        model = models[arch]()
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