Writing a dataset to FFCV format
================================

Datasets in FFCV are stored in a custom ``.beton`` format that allows for fast 
reading (see the _`making_dataloaders` section). 
We provide a utility function :meth:`ffcv.datasets.` for converting PyTorch datasets to
`.beton` files, as well as an example conversion script for CIFAR-10 and
ImageNet in :file:`scripts/TODO`.

In this tutorial, we'll briefly show how to turn any generic dataset into a
corresponding FFCV dataset. The main class we'll be using is the
:class:`ffcv.writer.DatasetWriter`:

.. code-block:: python 

    from ffcv.writer import DatasetWriter

We'll assume that you have a dataset in the form of a Python object implementing
both (a) the ``__getitem__`` function, returning a ``tuple`` of data, and 
(b) the ``__len__`` function, returning the number of examples in the dataset.
Valid examples include any `PyTorch dataset <TODO>`_, a Python ``list`` of
tuples, or even an ``N x 1 x d`` numpy array. For this example, we'll construct
a simple linear regression dataset that returns an input vector and its
corresponding label:

.. code-block:: python

    import numpy as np

    class LinearRegressionDataset:
        def __init__(self, N, d)
            self.X = np.randn(N, d)
            self.Y = np.randn(N)
        
        def __getitem__(self, idx):
            return (self.X[idx], self.Y[idx])
        
        def __len(self):
            return len(self.X)

    N, d = (100, 6)
    dataset = LinearRegressionDataset(N, d)

The class responsible for converting datasets to FFCV format is the
:class:`ffcv.writer.DatasetWriter`. The writer is
initialized with a dataset size, a path (where the `.beton` will be written),
and a dictionary mapping keys to *fields* (:class:`~ffcv.fields.Field`).
Each field corresponds to an element of the data tuple returned by our
dataset, and specifies how the element should be written to (and later, read
from) the FFCV dataset file. In our case, the dataset has two fields, one
for the (vector) input and the other for the corresponding (scalar) label.  
Both of these fields already have default implementations in FFCV, which we use
below: 

.. code-block:: python 

    from ffcv.fields import NDArrayField, FloatField

    writer = DatasetWriter(len(dataset), write_path, {
        'covariate': NDArrayField(shape=(d,), dtype=np.float32),
        'label': FloatField(),
    })

.. note:: 

    Starting in Python 3.6, dictionary keys are ordered, and the writer uses
    this order to match the given fields to the elements returned by the
    ``__getitem__`` function of the dataset. Make sure to provide
    the fields in the right order to avoid errors.

Beyond these two, FFCV provides a variety of built-in fields that make most
datasets easy to convert directly:

- :class:`~ffcv.fields.RGBImageField`, which handles images including (optional) compression
  and resizing,

- :class:`~ffcv.fields.IntField` and :class:`~ffcv.fields.ByteField`, which handle simple scalar fields

After constructing the writer, the only remaining step is to write the dataset:

.. code-block:: python

    with writer:
        writer.write_pytorch_dataset(my_dataset,
                                     num_workers=num_workers)

That's it! You are now ready to `Construct a loader <TODO>`_ for this dataset
and start training ML models!