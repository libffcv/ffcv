import os
from typing import List
from ffcv.pipeline.operation import Operation
from ffcv.transforms.mixup import ImageMixup, LabelMixup, MixupToOneHot
import torch as ch
from pathlib import Path
from torch.cuda.amp import GradScaler
import matplotlib
matplotlib.use('module://itermplot')

import time
from ffcv.transforms.ops import ToTorchImage
from trainer import Trainer
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, Convert, RandomHorizontalFlip
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from torchvision.transforms import Normalize
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf
from argparse import ArgumentParser
import numpy as np
from torchvision import models
import torch.optim as optim

# distributed imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


Section('model', 'model details').params(
    arch=Param(And(str, OneOf(models.__dir__())),
               'the architecture to use', required=True),
    antialias=Param(int, 'use blurpool or not', default=0)
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=224),
    max_res=Param(int, 'the maximum (starting) resolution', default=224),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=0),
    start_ramp=Param(int, 'when to start interpolating resolution', default=0)
)

Section('training', 'training hyper param stuff').params(
    step_ratio=Param(float, 'learning rate step ratio', default=0.1),
    step_length=Param(int, 'learning rate step length', default=30),
    lr_schedule_type=Param(OneOf(['step', 'linear']), 'step or linear schedule?',
                           required=True),
    eval_only=Param(int, 'eval only?', default=0),
    pretrained=Param(int, 'is pretrained?', default=0),
    bn_wd=Param(int, 'should momentum on bn', default=0)
)

Section('dist').enable_if(lambda cfg: cfg['training.distributed'] == 1).params(
    world_size=Param(int, 'number gpus', default=1),
    addr=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='12355')
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
DEFAULT_CROP_RATIO = 224/256

@param('training.lr')
@param('training.step_ratio')
@param('training.step_length')
def get_step_lr(epoch, lr, step_ratio, step_length):
    num_steps = epoch // step_length
    return step_ratio**num_steps * lr

@param('training.lr')
@param('training.epochs')
def get_linear_lr(epoch, lr, epochs):
    return np.interp([epoch], [0, epochs + 1], [lr, 0])[0]

class ImageNetTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setup_distributed()


    @param('dist.address')
    @param('dist.port')
    @param('dist.world_size')
    def setup_distributed(self, address, port, world_size):
        os.environ['MASTER_ADDR'] = address
        os.environ['MASTER_PORT'] = port

        dist.init_process_group("gloo", rank=self.gpu, world_size=world_size)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    @param('training.lr_schedule_type')
    def get_lr(self, epoch, lr_schedule_type):
        lr_schedules = {
            'linear': get_linear_lr,
            'step': get_step_lr
        }

        return lr_schedules[lr_schedule_type](epoch)

    # resolution tools
    @param('resolution.min_res')
    @param('resolution.max_res')
    @param('resolution.end_ramp')
    @param('resolution.start_ramp')
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        return min_res + int(interp[0] // 32) * 32

    @param('training.momentum')
    @param('training.optimizer')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    @param('training.bn_wd')
    def create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing, bn_wd):
        assert optimizer == 'sgd'

        if not bn_wd:
            all_params = list(self.model.named_parameters())
            bn_params = [v for k, v in all_params if ('bn' in k)]
            other_params = [v for k, v in all_params if not ('bn' in k)]
            param_groups = [{
                'params': bn_params,
                'weight_decay': 0.
            }, {
                'params': other_params,
                'weight_decay': weight_decay
            }]
        else:
            param_groups = [{
                'params': self.model.parameters(),
                'weight_decay': weight_decay
            }]

        self.optimizer = optim.SGD(param_groups, lr=1, momentum=momentum)
        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    @param('data.train_dataset')
    @param('training.batch_size')
    @param('training.mixup_alpha')
    @param('training.mixup_same_lambda')
    @param('data.num_workers')
    @param('training.eval_only')
    def create_train_loader(self, train_dataset, batch_size, mixup_alpha,
                            mixup_same_lambda, num_workers, eval_only):
        if eval_only:
            return []

        train_path = Path(train_dataset)
        assert train_path.is_file()
        self.decoder = RandomResizedCropRGBImageDecoder((224, 224))

        image_pipeline: List[Operation] = [self.decoder]
        if mixup_alpha:
            image_pipeline.append(ImageMixup(mixup_alpha, mixup_same_lambda))
        
        image_pipeline.extend([
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(ch.device('cuda:0'), non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            Normalize((IMAGENET_MEAN * 255).tolist(),
                      (IMAGENET_STD * 255).tolist()),
        ])

        label_pipeline: List[Operation] = [IntDecoder()]
        if mixup_alpha:
            label_pipeline.append(LabelMixup(mixup_alpha, mixup_same_lambda))

        label_pipeline.extend([
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device('cuda:0'), non_blocking=True)
        ])

        if mixup_alpha:
            label_pipeline.append(MixupToOneHot(1000))

        loader = Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.QUASI_RANDOM,
                        os_cache=True,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
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
                                         non_blocking=True),
                                ToTorchImage(),
                                Convert(ch.float16),
                                Normalize((IMAGENET_MEAN * 255).tolist(),
                                          (IMAGENET_STD * 255).tolist())
                            ],
                            'label': [IntDecoder(), ToTensor(), Squeeze(),
                                      ToDevice(ch.device('cuda:0'),
                                      non_blocking=True)]
                        })
        return loader

    @param('training.epochs')
    def train(self, epochs):
        for epoch in range(epochs):
            res = self.get_resolution(epoch)
            self.decoder.output_size = (res, res)
            train_loss = self.train_loop(epoch)

            extra_dict = {
                'train_loss': train_loss,
                'epoch': epoch
            }

            # if epoch % 5 == 0 or epoch == epochs - 1:
            self.eval_and_log(extra_dict)

        ch.save(self.model.state_dict(), self.log_folder / 'final_weights.pt')

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

        return stats

    @param('model.arch')
    @param('model.antialias')
    @param('training.pretrained')
    @param('training.distributed')
    def create_model_and_scaler(self, arch, antialias, pretrained, distributed):
        scaler = GradScaler()
        if not antialias:
            model = getattr(models, arch)(pretrained=pretrained)
        else:
            raise ValueError('dont do this eom')

        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)
        if distributed:
            model = DDP(model, device_ids=[self.gpu])
        return model, scaler

def make_trainer(gpu=0):
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    trainer = ImageNetTrainer(config, gpu=gpu)
    return trainer

# def execute(trainer, eval_only=False):
#     print(eval_only)
#     if not eval_only:
#         trainer.train()
#     else:
#         trainer.eval_and_log()

#     return trainer

@param('training.distributed')
@param('training.eval_only')
def exec(gpu, distributed, eval_only):
    trainer = make_trainer(gpu)
    if eval_only:
        trainer.eval_and_log()
    else:
        trainer.train()

    if distributed:
        print('was distributed')
        trainer.cleanup_distributed()

@param('training.distributed')
def main(distributed):
    if distributed:
        exec_distributed()
    else:
        exec()

@param('dist.world_size')
def exec_distributed(world_size):
    mp.spawn(exec, nprocs=world_size, join=True)

if __name__ == "__main__":
    main()