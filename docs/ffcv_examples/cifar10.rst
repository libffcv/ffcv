Training CIFAR-10 in 30 seconds on a single A100
================================================

In this example, we'll show how to use FFCV and the ResNet-9 architecture in
order to train a CIFAR-10 classifier to TODO% accuracy in TODO seconds.

Step 1: Create an FFCV-compatible CIFAR-10 dataset
==================================================

First, we'll use the :ref:`Writing datasets <Writing a dataset to FFCV format>`
guide to convert CIFAR-10 to FFCV format:

.. code-block:: python

    import torchvision

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

.. code-block:: python

    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]

    loaders = {}
    for name in ['train', 'test']:
        # Create loaders
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2),
                Cutout(8, tuple(map(int, CIFAR_MEAN))),
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        loaders[name] = Loader(f'/tmp/cifar_{name}.beton', batch_size=512, num_workers=8,
                                order=OrderOption.RANDOM, drop_last=(name == 'train'),
                                pipelines={'image': image_pipeline, 'label': label_pipeline})


Step 3: Define model architecture and optimization parameters
=============================================================

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
    )
    model = model.to(memory_format=ch.channels_last).cuda()




Step 4: Train the model
=======================



