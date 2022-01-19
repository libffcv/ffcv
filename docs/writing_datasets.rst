Writing a dataset to FFCV format
================================

Datasets in FFCV are stored in a custom ``.beton`` format that allows for fast
reading (see the :ref:`Making an FFCV dataloader <Making an FFCV dataloader>` section).

Such files can be generated using the class :class:`ffcv.writer.DatasetWriter` from two potential sources:

- **Indexable objects**:
  They need to implement ``__len__`` and a ``__getitem__`` function
  returning the data associated to a sample as a tuple/list (of any length).
  Examples of this kind of dataset include but are not limited to:
  ``torch.utils.data.Dataset``, ``numpy.ndarray``, or even Python lists.
- **Webdataset** (`Github <https://github.com/webdataset/webdataset>`_):
  This allows users to integrate large scale and/or remote datasets into FFCV easily.

In this tutorial, we will show how to handle datasets from these two categories.
Additionally, in the folder ``/examples`` of our `repository <https://github.com/libffcv/ffcv>`_ we also include a
conversion script illustrating the conversion of `CIFAR-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ and `ImageNet <https://www.image-net.org>`_ from their PyTorch counterparts.

The first step is to include the following class into your script:

.. code-block:: python

    from ffcv.writer import DatasetWriter

Indexable Dataset
-----------------

For this example, we'll construct a simple linear regression dataset that
returns an input vector and its corresponding label:

.. code-block:: python

    import numpy as np

    class LinearRegressionDataset:
        def __init__(self, N, d):
            self.X = np.randn(N, d)
            self.Y = np.randn(N)

        def __getitem__(self, idx):
            return (self.X[idx], self.Y[idx])

        def __len__(self):
            return len(self.X)

    N, d = (100, 6)
    dataset = LinearRegressionDataset(N, d)

.. note ::
    The class ``LinearRegressionDataset`` implements the interface required to be a
    ``torch.utils.data.Dataset`` so one could use any PyTorch Dataset instead of our
    toy example here.

The class responsible for converting datasets to FFCV format is the
:class:`ffcv.writer.DatasetWriter`. The writer takes in:

- A path, where the ``.beton`` will be written
- A dictionary mapping keys to *fields* (:class:`~ffcv.fields.Field`).

Each field corresponds to an element of the data tuple returned by our
dataset, and specifies how the element should be written to (and later, read
from) the FFCV dataset file.  In our case, the dataset has two fields, one
for the (vector) input and the other for the corresponding (scalar) label.
Both of these fields already have default implementations in FFCV, which we use
below:

.. code-block:: python

    from ffcv.fields import NDArrayField, FloatField

    writer = DatasetWriter(write_path, {
        'covariate': NDArrayField(shape=(d,), dtype=np.dtype('float32')),
        'label': FloatField(),

    }, num_workers=16)
.. note::

    Starting in Python 3.6, dictionary keys are ordered, and :class:`~ffcv.writer.DatasetWriter` uses
    this order to match the given fields to the elements returned by the
    ``__getitem__`` function of the dataset. Make sure to provide
    the fields in the right order to avoid errors.


After constructing the writer, the only remaining step is to write the dataset:

.. code-block:: python

    writer.from_indexed_dataset(my_dataset)

Webdataset
----------

For this second example we will assume that you have access to a
``webdataset`` version of ImageNet (or similar) dataset, and that all the
shards are in a folder called ``FOLDER``.

In order to perform the conversion to a ``.beton`` file, we first need to
collect the list of shards. This can be simply done with ``glob``:

.. code-block:: python

    from glob import glob
    from os import path

    my_shards = glob(path.join(FOLDER, '*'))

Internally, FFCV will split the shards between the available workers.
However, each worker still needs to know how to decode a given shard. This is done
by defining a pipeline (very similar to how one would use a ``webdataset`` for training):

.. code-block:: python

    def pipeline(dataset):
        return dataset.decode('rgb8').to_tuple('jpg:png;jpeg cls')

Since FFCV expects images in the numpy ``uint8`` format, we use the parameter ``'rgb8'``
of ``webdataset`` to decode the images. We then convert the dictionary to a tuple
that FFCV will be able to process.

We now just have to glue everything together:


.. code-block:: python

    from ffcv.fields import RGBImageField, IntField

    writer = DatasetWriter(write_path, {
        'image': RGBImageField()
        'label': IntField(),

    }, num_workers=40)

    writer.from_webdataset(my_shards, pipeline)


Fields
------

Beyond the examples used above, FFCV supports a variety of built-in fields that make it easy to directly convert most datasets. We review them below:

- :class:`~ffcv.fields.RGBImageField`: Handles images including (optional) compression
  and resizing. Pass in a PyTorch Tensor.
- :class:`~ffcv.fields.IntField` and :class:`~ffcv.fields.FloatField`: Handle simple scalar fields.
  Pass in ``int`` or ``float``.
- :class:`~ffcv.fields.BytesField`: Stores byte arrays of variable length. Pass in ``numpy`` byte array.
- :class:`~ffcv.fields.JSONField`: Encodes a JSON document. Pass in ``dict`` that can be JSON-encoded.


That's it! You are now ready to :ref:`construct loaders<Making an FFCV dataloader>` for this dataset
and start loading the data.






