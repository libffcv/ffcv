Training CIFAR-10 in 30 seconds on a single A100
================================================

In this example, we'll show how to use FFCV and the ResNet-9 architecture in
order to train a CIFAR-10 classifier to TODO% accuracy in TODO seconds on a A100 GPU.

First, we import ``torch`` and necessary components from ``ffcv``.

.. code-block:: python

    import torch as ch
    from typing import List
    import torchvision

    from ffcv.fields import IntField, RGBImageField
    from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
    from ffcv.loader import Loader, OrderOption
    from ffcv.pipeline.operation import Operation
    from ffcv.transforms import RandomHorizontalFlip, Cutout, \
        RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
    from ffcv.transforms.common import Squeeze
    from ffcv.writer import DatasetWriter


Step 1: Create an FFCV-compatible CIFAR-10 dataset
==================================================

First, we'll use the :ref:`Writing datasets <Writing a dataset to FFCV format>`
guide to convert CIFAR-10 to FFCV format:

.. code-block:: python

    datasets = {
        'train': torchvision.datasets.CIFAR10('/tmp', train=True, download=True),
        'test': torchvision.datasets.CIFAR10('/tmp', train=False, download=True)
    }

    for (name, ds) in datasets.items():
    writer = DatasetWriter(f'/tmp/cifar_{name}.beton', {
        'image': RGBImageField(),
        'label': IntField()
    })
    writer.from_indexed_dataset(ds)


Step 2: Create the train and test loaders
=========================================

Next, we construct FFCV dataloaders from the ``.beton`` dataset file created above.
(See :ref:`Making an FFCV dataloader` for general how to.)

For training set, we use a set of standard data augmentations: random horizontal flip,
random translation, and Cutout.

.. code-block:: python

    # Note that statistics are wrt to uin8 range, [0,255].
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]

    BATCH_SIZE = 512

    loaders = {}
    for name in ['train', 'test']:
        # Create loaders
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2),
                Cutout(8, tuple(map(int, CIFAR_MEAN))), # Note Cutout is done before normalization.
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        loaders[name] = Loader(f'/tmp/cifar_{name}.beton',
                                batch_size=BATCH_SIZE,
                                num_workers=8,
                                order=OrderOption.RANDOM,
                                drop_last=(name == 'train'),
                                pipelines={'image': image_pipeline,
                                           'label': label_pipeline})


Step 3: Define model architecture and optimization parameters
=============================================================

We now define the model. We use a custom ResNet-9 architecture from KakaoBrain
(https://github.com/wbaek/torchskeleton).

.. code-block:: python

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
                             kernel_size=kernel_size, stride=stride, padding=padding,
                             groups=groups, bias=False),
                ch.nn.BatchNorm2d(channels_out),
                ch.nn.ReLU(inplace=True)
        )

    NUM_CLASSES = 10
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
        ch.nn.Linear(128, NUM_CLASSES, bias=False),
        Mul(0.2)
    )
    model = model.to(memory_format=ch.channels_last).cuda()

Note the ``ch.channels_last`` option when we put the model on GPU.


Next, we define the optimizer and hyperparameters.
We use standard SGD on the cross entropy loss with a cyclic learning rate schedule (single peak).

.. code-block:: python
    
    import numpy as np
    from torch.cuda.amp import GradScaler, autocast
    from torch.nn import CrossEntropyLoss
    from torch.optim import SGD, lr_scheduler

    EPOCHS = 24

    opt = SGD(model.parameters(), lr=.5, momentum=0.9, weight_decay=5e-4)
    iters_per_epoch = 50000 // BATCH_SIZE
    lr_schedule = np.interp(np.arange((EPOCHS+1) * iters_per_epoch),
                            [0, 5 * iters_per_epoch, EPOCHS * iters_per_epoch],
                            [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)



Step 4: Train and evaluate the model!
=====================================

Finally, we're ready to train our model.

.. code-block:: python

    from tqdm import tqdm

    for ep in range(EPOCHS):
        for ims, labs in tqdm(loaders['train']):
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

    model.eval()
    with ch.no_grad():
        total_correct, total_num = 0., 0.
        for ims, labs in tqdm(loaders['test']):
            with autocast():
                out = (model(ims) + model(ch.fliplr(ims))) / 2. # Test-time augmentation
                total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                total_num += ims.shape[0]

        print(f'Accuracy: {total_correct / total_num * 100:.1f}%')


Wrapping up
===========

In this tutorial, we used FFCV to train a CIFAR-10 classifier to TODO% accuracy in TODO seconds.
For more detailed benchmarks, see :ref:`Benchmarks`.
