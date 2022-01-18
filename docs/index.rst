.. ffcv documentation master file, created by
   sphinx-quickstart on Sun Nov  7 17:08:11 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FFCV's documentation!
================================

View `Homepage <https://ffcv.io>`_ or on `GitHub <https://github.com/libffcv/ffcv>`_.

Install ``ffcv``:

.. code-block:: bash

   conda create -y -n ffcv python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c pytorch -c conda-forge
   conda activate ffcv
   pip install ffcv


Introduction
------------

``ffcv`` is a drop-in data loading system that dramatically increases data throughput in model training:

- Train an ImageNet model on one GPU in 35 minutes (98¢/model on AWS)
- Train a CIFAR-10 model on one GPU in 36 seconds (2¢/model on AWS)
- Train a ``$YOUR_DATASET`` model ``$REALLY_FAST`` (for ``$WAY_LESS``)

Keep your training algorithm the same, just replace the data loader! Look at these speedups:

.. image:: /_static/headline.svg
  :width: 100%

|

With ``ffcv``, we enable significantly faster training:

.. image:: /_static/perf_scatterplot.svg

|
See :ref:`ImageNet Benchmarks` for further benchmark details.

See the :ref:`Features` section below for a more detailed glance at what FFCV can do.


Tutorials
---------

We provide a walk-through of basic usage, a performance guide, complete
examples (including advanced customizations), as well as detailed benchmarks on
ImageNet.

.. toctree::
   quickstart
   basics
   performance_guide
   examples
   benchmarks
   :maxdepth: 2

Features
--------

Computer vision or not, FFCV can help make training faster in a variety of
resource-constrained settings!
Our :ref:`Performance Guide` has a more detailed account of the ways in which
FFCV can adapt to different performance bottlenecks.


- **Plug-and-play with any existing training code**: Rather than changing
  aspects of model training itself, FFCV focuses on removing *data bottlenecks*,
  which turn out to be a problem everywhere from neural network training to
  linear regression. This means that:

  - FFCV can be introduced into any existing training code in just a few
    lines of code (e.g., just swapping out the data loader and optionally the
    augmentation pipeline);
  - you don't have to change the model itself to make it faster (e.g., feel
    free to analyze models *without* CutMix, Dropout, momentum scheduling, etc.);
  - FFCV can speed up a lot more beyond just neural network training---in
    fact, the more data-bottlenecked the application (e.g., linear regression,
    bulk inference, etc.), the faster FFCV will make it!

  See our :ref:`Getting started` guide, :ref:`Examples`, and
  `code examples <https://github.com/libffcv/ffcv/tree/main/examples>`_
  to see how easy it is to get started!
- **Fast data processing without the pain**: FFCV automatically handles data
  reading, pre-fetching, caching, and transfer between devices in an extremely
  efficiently way, so that users don't have to think about it.
- **Automatically fused-and-compiled data processing**: By either using
  `pre-written <https://docs.ffcv.io/api/transforms.html>`_ FFCV transformations
  or
  :ref:`easily writing custom ones <Fast custom image transforms>`,
  users can
  take advantage of FFCV's compilation and pipelining abilities, which will
  automatically fuse and compile simple Python augmentations to machine code
  using `Numba <https://numba.pydata.org/>`_, and schedule them asynchronously to avoid
  loading delays.
- **Load data fast from RAM, SSD, or networked disk**: FFCV exposes
  user-friendly options that can be adjusted based on the resources
  available. For example, if a dataset fits into memory, FFCV can cache it
  at the OS level and ensure that multiple concurrent processes all get fast
  data access. Otherwise, FFCV can use fast process-level caching and will
  optimize data loading to minimize the underlying number of disk reads. See
  :ref:`The Bottleneck Doctor <The Bottleneck Doctor>`
  guide for more information.
- **Training multiple models per GPU**: Thanks to fully asynchronous
  thread-based data loading, you can now interleave training multiple models on
  the same GPU efficiently, without any data-loading overhead. See
  :ref:`this guide <Tuning Guide>` for more info.
- **Dedicated tools for image handling**: All the features above are
  equally applicable to all sorts of machine learning models, but FFCV also
  offers some vision-specific features, such as fast JPEG encoding and decoding,
  storing datasets as mixtures of raw and compressed images to trade off I/O
  overhead and compute overhead, etc. See the :ref:`Working with images <Working with Image Data in FFCV>` guide for
  more information.


API Reference
-------------

.. toctree::
   api_reference


Citation
--------
If you use this library in your research, cite it as
follows:

.. code-block:: bibtex

   @misc{leclerc2022ffcv,
      author = {Guillaume Leclerc and Andrew Ilyas and Logan Engstrom and Sung Min Park and Hadi Salman and Aleksander Madry},
      title = {ffcv},
      year = {2022},
      howpublished = {\url{https://github.com/libffcv/ffcv/}},
      note = {commit xxxxxxx}
   }

*(Have you used the package and found it useful? Let us know!)*.


Contributors
-------------
- `Guillaume Leclerc <https://twitter.com/gpoleclerc>`_
- `Andrew Ilyas <https://twitter.com/andrew_ilyas>`_
- `Logan Engstrom <https://twitter.com/logan_engstrom>`_
- `Sam Park <https://twitter.com/smsampark>`_
- `Hadi Salman <https://twitter.com/hadisalmanX>`_


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


