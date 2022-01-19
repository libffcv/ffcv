Making an FFCV dataloader
=========================

After :ref:`writing an FFCV dataset <Writing a dataset to FFCV format>`, we are
ready to start loading data (and training models)! We'll continue using the same
regression dataset as the previous guide, and we'll assume that the dataset has
been written to ``/path/to/dataset.beton``.

In order to load the dataset that we've written, we'll need the
:class:`ffcv.loader.Loader` class (which will do most of the heavy lifting), and
a set of *decoders* corresponding to the fields present in the dataset (so in
our case, we will use the :class:`~ffcv.fields.decoders.FloatDecoder` and
:class:`~ffcv.fields.decoders.NDArrayDecoder` classes):

.. code-block:: python

    from ffcv.loader import Loader, OrderOption
    from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder

Our first step is instantiating the :class:`~ffcv.loader.Loader` class:

.. code-block:: python

  loader = Loader('/path/to/dataset.beton',
                  batch_size=BATCH_SIZE,
                  num_workers=NUM_WORKERS,
                  order=ORDERING,
                  pipelines=PIPELINES)

In order to create a loader, we need to specify a path to the FFCV dataset,
batch size, number of workers, as well as two less standard arguments, ``order``
and ``pipelines``, which we discuss below:

Dataset ordering
''''''''''''''''
The ``order`` option in the loader initialization is similar to `PyTorch DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_'s ``shuffle`` option, with some additional options. This argument
takes an ``enum`` provided by :class:`ffcv.loader.OrderOption`:

.. code-block:: python

  from ffcv.loader import OrderOption

  # Truly random shuffling (shuffle=True in PyTorch)
  ORDERING = OrderOption.RANDOM

  # Unshuffled (i.e., served in the order the dataset was written)
  ORDERING = OrderOption.SEQUENTIAL

  # Memory-efficient but not truly random loading
  # Speeds up loading over RANDOM when the whole dataset does not fit in RAM!
  ORDERING = OrderOption.QUASI_RANDOM

Pipelines
'''''''''
The ``pipeline`` option in :class:`~ffcv.loader.Loader` specifies the dataset and
tells the loader what fields to read, how to read them, and what operations to
apply on top. Specifically, a pipeline is a key-value dictionary where the key
matches the one used in :ref:`writing the dataset <Writing a dataset to FFCV format>`, and the value is a
sequence of operations to perform. The operations must start with a
:class:`ffcv.fields.decoders.Decoder` object corresponding to that field followed by a
sequence of *transforms*.
For example, the following pipeline reads the fields and then converts each one
to a PyTorch tensor:

.. code-block:: python

  from ffcv.transforms import ToTensor

  PIPELINES = {
    'covariate': [NDArrayDecoder(), ToTensor()],
    'label': [FloatDecoder(), ToTensor()]
  }

This is already enough to start loading data, but pipelines are also our
opportunity to apply fast pre-processing to the data through a series of
transformations---transforms are automatically compiled to machine code at runtime
and, for GPU-intensive applications like training neural networks, can reduce
any additional training overhead.

.. note::

  In fact, declaring field pipelines is optional: for any field that exists
  in the dataset file without a corresponding pipeline specified in the
  ``pipelines`` dictionary,  the :class:`~ffcv.loader.Loader` will default to
  the bare-bones pipeline above, i.e., first a decoder
  then a conversion to PyTorch tensor. (You can force FFCV to explicitly *not*
  load a field by adding a corresponding ``None`` entry to the ``pipelines``
  dictionary.)

  If the entire ``pipelines`` argument is
  unspecified, this bare-bones pipeline will be applied to all fields.

Transforms
"""""""""""

There are three easy ways to specify transformations in a pipeline:

- A set of standard transformations in the
  :mod:`ffcv.transforms` module. These include standard image data augmentations such as :class:`~ffcv.transforms.RandomHorizontalFlip` and :class:`~ffcv.transforms.Cutout`.

- Any subclass of ``torch.nn.Module``: FFCV automatically converts them into an operation.

- Custom transformations: you can implement your own by subclassing
  :class:`ffcv.transforms.Operation`, as discussed in the
  :ref:`Making custom transforms <Fast custom image transforms>` guide.

The following shows an example of a full pipeline for a vector field starts with the field decoder,
:class:`~ffcv.fields.decoders.NDArrayDecoder`, followed by conversion to ``torch.Tensor``, and a custom transform implemented as a :class:`torch.nn.Module` that adds Gaussian noise to each vector:

.. code-block:: python

    class AddGaussianNoise(ch.nn.Module):
        def __init__(self, scale=1):
            super(AddGaussianNoise, self).__init__()
            self.scale = scale

        def forward(self, x):
            return x + ch.randn_like(x) * self.scale

    pipeline: List[Operation] = [
        NDArrayDecoder(),
        ToTensor(),
        AddGaussianNoise(0.1)
    ]


For an example of a different field, this could be a pipeline for an :class:`~ffcv.fields.RGBImageField`:

.. code-block:: python

    image_pipeline: List[Operation] = [
        SimpleRGBImageDecoder(),
        RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(.4,.4,.4),
        RandomTranslate(padding=2),
        ToTensor(),
        ToDevice('cuda:0', non_blocking=True),
        ToTorchImage(),
        Convert(ch.float16),
        torchvision.transforms.Normalize(MEAN, STD), # Normalize using image statistics
    ])


Putting together
''''''''''''''''

Back to our running linear regression dataset example, in summary the final loader can be constructed as follows:

.. code-block:: python

  loader = Loader('/path/to/dataset.beton',
                  batch_size=BATCH_SIZE,
                  num_workers=NUM_WORKERS,
                  order=OrderOption.RANDOM,
                  pipelines={
                    'covariate': [NDArrayDecoder(), ToTensor(), AddGaussianNoise(0.1)],
                    'label': [FloatDecoder(), ToTensor()]
                  })




Other options
'''''''''''''

You can also specify the following additional options when constructing an :class:`ffcv.loader.Loader`:

- ``os_cache``: If True, the entire dataset is cached
- ``distributed``: For training on :ref:`multiple GPUs<Scenario: Multi-GPU training (1 model, multiple GPUs)>`
- ``seed``: Specify the random seed for batch ordering
- ``indices``: Provide indices to load a subset of the dataset
- ``custom_fields``: For specifying decoders for fields with custom encoders
- ``drop_last``: If True, drops the last non-full batch from each iteration
- ``batches_ahead``: Set the number of batches prepared in advance. Increasing it absorbs variation in processing time to make sure the training loop does not stall for too long to process batches. Decreasing it reduces RAM usage.
- ``recompile``: Recompile every iteration. Useful if you have transforms that change their behavior from epoch to epoch, for instance code that uses the shape as a compile time param. (But if they just change their memory usage, e.g., the resolution changes, it's not necessary.)


More information
''''''''''''''''

For information on available transforms and the :class:`~ffcv.loader.Loader` class, see our :ref:`API Reference`.

For examples of constructing loaders and using them, see the tutorials :ref:`Training CIFAR-10 in 36 seconds on a single A100`
and :ref:`Large-Scale Linear Regression`.
