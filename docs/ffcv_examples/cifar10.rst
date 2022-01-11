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

