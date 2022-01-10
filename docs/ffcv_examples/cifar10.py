from ffcv.loader.loader import OrderOption
from ffcv.transforms.common import Squeeze
from ffcv.transforms.crop import RandomTranslate
from ffcv.transforms.ops import Convert, ToDevice, ToTensor, ToTorchImage
from torchvision.datasets import CIFAR10
from torchvision import transforms
from ffcv.writer import DatasetWriter
from ffcv.loader import Loader
from ffcv.transforms import RandomHorizontalFlip, Cutout
from tqdm import tqdm
import torch as ch

from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.fields import IntField, RGBImageField

datasets = {
    'train': CIFAR10('/tmp', train=True, download=True),
    'test': CIFAR10('/tmp', train=False, download=True)
}

for (name, ds) in datasets.items():
    writer = DatasetWriter(f'/tmp/cifar_{name}.beton', {
        'image': RGBImageField(),
        'label': IntField()
    })

    writer.from_indexed_dataset(ds)

# Create loaders
label_pipeline = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]

import torch.nn.functional as F

class AffineAugmentation(ch.nn.Module):
    '''
    Reflection padding + random crop + image reflections
    '''
    def __init__(self, padding):
        super(AffineAugmentation, self).__init__()
        self.padding = ch.arange(-padding, padding + 1).half().cuda()

    def forward(self, x):
        #unif = ch.empty(x.shape[0], 2, device='cuda', dtype=ch.int).int()
        tx_size = (x.shape[0], 2)
        #tx = np.random.choice(self.padding/(x.shape[3]/2), size=tx_size, replace=True)
        tx_ix = ch.randint(len(self.padding), tx_size, dtype=ch.long)
        tx = self.padding[tx_ix] / (x.shape[3]/2)
        mat = ch.zeros(x.shape[0], 2, 3, device='cuda', dtype=ch.half)

        reflects = ch.empty(x.shape[0], device='cuda', dtype=ch.half) \
            .bernoulli_().mul_(2.).add_(-1.)

        mat[:,0,0] = reflects
        mat[:,1,1] = 1.
        mat[:,:,2] = tx
        x = F.grid_sample(x, F.affine_grid(mat, x.shape, align_corners=False),
                          align_corners=False, mode='nearest',
                          padding_mode='reflection')
        return x


CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]
train_loader = Loader('/tmp/cifar_train.beton', batch_size=512, num_workers=8,
                order=OrderOption.RANDOM, pipelines={
                    'image': [
                        SimpleRGBImageDecoder(),
                        RandomHorizontalFlip(),
                        RandomTranslate(padding=2),
                        Cutout(8, tuple(map(int, CIFAR_MEAN))),
                        ToTensor(),
                        ToDevice('cuda:0', non_blocking=True),
                        ToTorchImage(),
                        Convert(ch.float16),
                        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                    ], 'label': label_pipeline})
test_loader = Loader('/tmp/cifar_test.beton', batch_size=512, num_workers=8,
                order=OrderOption.RANDOM, pipelines={
                    'image': [
                        SimpleRGBImageDecoder(),
                        ToTensor(),
                        ToDevice('cuda:0', non_blocking=True),
                        ToTorchImage(),
                        Convert(ch.float16),
                        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                    ], 'label': label_pipeline})

# Model (from KakaoBrain: https://github.com/wbaek/torchskeleton)

class Mul(ch.nn.Module):
    def __init__(self, weight):
       super(Mul, self).__init__()
       self.weight = weight
    def forward(self, x): return x * self.weight

class Flatten(ch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Residual(ch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return ch.nn.Sequential(
            ch.nn.Conv2d(channels_in, channels_out,
                            kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            ch.nn.BatchNorm2d(channels_out),
            ch.nn.ReLU(inplace=True)
    )

num_class = 10
model = ch.nn.Sequential(
    conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
    conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
    Residual(ch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
    conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
    ch.nn.MaxPool2d(2),
    Residual(ch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
    conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
    ch.nn.AdaptiveMaxPool2d((1, 1)),
    Flatten(),
    ch.nn.Linear(128, num_class, bias=False),
    Mul(0.2)
).to(memory_format=ch.channels_last).cuda()

# Make optimizer and schedule
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD, lr_scheduler
from torch.nn import CrossEntropyLoss
import numpy as np

opt = SGD(model.parameters(), lr=.5, momentum=0.9, weight_decay=5e-4)
iters_per_epoch = 50000 // 512
lr_schedule = np.interp(np.arange(25 * iters_per_epoch), [0, 5 * iters_per_epoch, 24 * iters_per_epoch], [0, 1, 0])
scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
scaler = GradScaler()
loss_fn = CrossEntropyLoss(label_smoothing=0.1)

test_freq = 2
for ep in range(48):
    is_train = (ep % test_freq !=  test_freq - 1)
    model.train(is_train)
    ch.set_grad_enabled(is_train)
    total_loss, total_correct, total_num = 0., 0., 0.
    for ims, labs in tqdm(train_loader if is_train else test_loader):
        # print(ims.shape, ims)
        opt.zero_grad(set_to_none=True)
        with autocast():
            out = model(ims) if is_train else \
                    model(ims) + model(ch.fliplr(ims))
            loss = loss_fn(out, labs)
            if not is_train:
                total_correct += out.argmax(1).eq(labs).cpu().float().sum().item() 
                total_loss += loss.detach().cpu().item() * ims.shape[0]
                total_num += ims.shape[0]

        if is_train:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
    if not is_train:
        print(f'Epoch {ep} | Loss: {total_loss / total_num} | Accuracy: {total_correct / total_num:.3f}')