import torch as ch
import torchvision
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

Section('model', 'model details').params(
    arch=Param(And(str, OneOf(['resnet9', 'resnet18'])), 'the architecture to use', required=True)
)

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
            Cutout(8),
            RandomHorizontalFlip(0.5),
            Collate(),
            ToTensor(),
            ToTorchImage(),
            ToDevice(ch.device('cuda:0')),
            Convert(ch.float16),
            Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                      std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                      inplace=True).to('cuda:0')
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
            ToTorchImage(),
            ToDevice('cuda:0'),
            Convert(ch.float16),
            Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                      std=[0.229 * 255, 0.224 * 255, 0.225 * 255]).to('cuda:0')
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
        model = torchvision.models.__dict__[arch]()
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
    args = config.get()
    hyper_params = {'.'.join(k): config[k] for k in config.entries.keys()}
    trainer = CIFARTrainer(hyper_params)
    trainer.train()