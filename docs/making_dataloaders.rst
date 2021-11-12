Making an FFCV dataloader
=========================

After `Writing an FFCV dataset <TODO>`_, we are ready to start loading data (and
training models)! We'll continue using the same regression dataset as the
previous guide, and we'll assume that the dataset has been written to
``/path/to/data.beton``.

In order to load the dataset that we've written, we'll need the
:class:`ffcv.loader.Loader` class (which will do most of the heavy lifting), and
a set of *decoders* corresponding to the fields present in the dataset (so in
our case, we will use the :class:`~ffcv.fields.decoders.FloatDecoder` and
:class:`~ffcv.fields.decoders.NDArrayDecoder` classes):

.. code-block:: python

    from ffcv.loader import Loader, OrderOption
    from ffcv.fields.decoders import SimpleRGBImageDecoder, FloatDecoder

In order to create a loader, we need to specify a path to the dataset, a batch
size, number of workers, as well as the following less familiar options:

- *Ordering*: dataset ordering is determined by the ``order`` parameter, which
  accepts either :attr:`ffcv.loader.OrderOption.RANDOM` for random ordering,
  :attr:`ffcv.loader.OrderOption.SEQUENTIAL` for sequential (non-shuffled)
  ordering, or :attr:`ffcv.loader.OrderOption.QUASIRANDOM`, which [TODO].

.. code-block:: python

    loader = Loader('/path/to/dataset.beton',
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    order=OrderOption.RANDOM,