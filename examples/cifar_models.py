import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
ch = torch

class Mul(torch.nn.Module):
    def __init__(self, weight):
       super(Mul, self).__init__()
       self.weight = weight

    def forward(self, x):
       return x * self.weight

class Flatten(torch.nn.Module):
    def forward(self, x):
       return x.view(x.size(0), -1)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1,
            groups=1, bn=True, activation=True, bn_weight_init=1.,
            bn_bias_init=0.):
    op = [
            torch.nn.Conv2d(channels_in, channels_out,
                            kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
    ]
    if bn:
        bn_layer = torch.nn.BatchNorm2d(channels_out)
        op.append(bn_layer)
    if activation:
        op.append(torch.nn.ReLU(inplace=True))
    return torch.nn.Sequential(*op)

class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)

class Cutout(ch.nn.Module):
    '''
    Cutout; height, width are h/w of boxes
    '''
    def __init__(self, height, width, groups):
        super().__init__()
        self.height = height
        self.width = width
        self.groups = groups
        self.filler = ch.tensor((0.4914, 0.4822, 0.4465))
        self.filler = self.filler.half().cuda()[..., None, None]

    def forward(self, image):
        bs, _, h, w = image.shape
        y = ch.randint(h+1-self.height, [self.groups], dtype=ch.long)
        x = ch.randint(w+1-self.width, [self.groups], dtype=ch.long)

        chunk_size = bs // self.groups + 1
        for i, (xx, yy) in enumerate(zip(x, y)):
            start, end = i * chunk_size, (i + 1) * chunk_size
            image[start:end, :, yy:yy+self.height, xx:xx+self.width] = self.filler

        return image

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

def build_network(num_class=10, w=1):
    return torch.nn.Sequential(
        conv_bn(3, 64*w, kernel_size=3, stride=1, padding=1),
        conv_bn(64*w, 128*w, kernel_size=5, stride=2, padding=2),
        # torch.nn.MaxPool2d(2),

        Residual(torch.nn.Sequential(
            conv_bn(128*w, 128*w),
            conv_bn(128*w, 128*w),
        )),

        conv_bn(128*w, 256*w, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),

        Residual(torch.nn.Sequential(
            conv_bn(256*w, 256*w),
            conv_bn(256*w, 256*w),
        )),

        conv_bn(256*w, 128*w, kernel_size=3, stride=1, padding=0),

        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128*w, num_class, bias=False),
        Mul(0.2)
    )

models = {
    'resnet9': build_network,
    # 'resnet18': ResNet18
}