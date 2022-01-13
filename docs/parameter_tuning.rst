Tuning Guide
=============

FFCV is a generic library and achieving the best performance for a particular application may requiring tuning some parameters to optimize to the paritcular system and resource environment.

In order to help users with that task, we consider a couple of common use cases and provide appropriate recommendations for setting parameters.

Scenario: Small dataset
-----------------------

If the dataset you are working on is small or if you are lucky enough to have a machine with large amounts of RAM, we recommend the following settings for :class:`ffcv.loader.Loader`:

- Use ``os_cache=True``. The first epoch will be a little bit slower as the operating system is not able to pre-fetch the data as well but once the dataset has been completely cached in RAM then it will read directly from there with no overhead.

- Set ``order`` to `OrderOption.RANDOM` or `OrderOption.RANDOM``. They should both perform very similarly (`QUASI_RANDOM` might be marginally better).


Scenario: Large scale datasets
------------------------------

If your dataset is too large to be cached on the machine we recommend:

- Use ``os_cache=False``. Since the data can't be cached, FFCV will have to read it over and over. Having FFCV take over the operating system for caching is beneficial as it knows in advance the which samples will be needed in the future and can load them ahead of time.
- For ``order``, we recommend using the ``QUASI_RANDOM`` traversal order if you need randomness but perfect uniform sampling isn't mission critical. This will optimize the order to minimize the reads on the underlying storage while maintaining very good randomness properties. If you have experience with the ``shuffle()`` function of ``webdataset`` and the quality of the randomness wasn't sufficient, we still suggest you give ``QUASI_RANDOM`` a try as it should be significantly better.


Scenario: Multi-GPU training (1 model, multiple GPUs)
-----------------------------------------------------

FFCV's :class:`~ffcv.loader.Loader` class offers a flag ``distributed`` that will make the loader behave similarly to the PyTorch's ``DistributedSampler`` used in a ``DataLoader``. If that's what your code is using, switching to FFCV should just be a matter of replacing the data loader.

FFCV should also work fine with PyTorch's ``DataParallel`` wrapper but we agree with the developers and recommend you use ``DistributedDataParallel`` with FFCV's ``distributed`` flag enabled.

The same recommendations above related to dataset size still apply here, but we emphasize that ``os_cache=True`` is particularly beneficial in this scenario. Indeed, as multiple processes will access the same dataset, having the caching at the OS level allows for data sharing between them, reducing overall memory consumption.

.. note ::
    `QUASI_RANDOM` isn't currently supported with ``distributed=True``. While this is technically possible to implement the team hasn't invested the necessary time yet. We also welcome pull requests.

We encourage users to try different values for the ``num_workers`` parameters. As FFCV is usually very CPU resource efficient it is sometimes beneficial to use fewer workers to avoid scheduling and cache inefficiencies.

Scenario: Grid search (1 model per GPU)
---------------------------------------

This use case is similar to the previous. One should still have one process per GPU and if training all models on the same dataset, ``os_cache=True`` is preferred to allow cache sharing between the jobs. Note that if the dataset is bigger than the amount of main memory, ``os_cache=False`` might still perform better and we encourage users to try both.

Scenario: Extreme grid search (2+ models per GPU)
--------------------------------------------------

Unlike other solutions, FFCV is thread based and not process based. As a result, users are able to train multiple models on a single GPU. This is particularly useful for small models that can't leverage the compute power of powerful GPUs. To do so users have to do the following:

- Run a **single** process per GPU
- The main thread of that process should start one thread for each model will be trained concurrently
- Each thread creates its own FFCV :class:`~ffcv.loader.Loader` and model and trains normally
- As for regular Grid search, ``os_cache=True`` is mostly the best choice here, but it doesn't hurt to try disabling it for very large scale datasets

.. warning ::
    It is a common mistake to assume that running multiple processes on the same GPU will improve speed. For security reasons and unless Nvidia MPS service is enabled, a GPU can only be used by a single process at a time. If you run more processes, GPU time will be shared between them but they will never run concurrently.

.. note ::
   We have experienced some CUDNN bugs while running multiple models on the same GPU. It seems to originate from scheduling concurrently multiple BatchNorm layers. If you encounter that issue a simple fix is to put a lock around the forward pass of your models. This will make sure that no two forward pass is scheduled concurrently. This shouldn't impact performance too much as CUDA calls are asynchronous anyway.

Summary: The Bottleneck Doctor
==============================
To summarize the scenarios above, we provide a map from a type of *system
bottleneck* to the FFCV options that will help get the most performance out of
your system:  

Disk-read bottlenecks 
---------------------
What if your GPUs sit idle from low disk or throughput?
Maybe you're reading from a networked drive, maybe you have too many GPUs;
either way, try:

- If your dataset fits in memory, use *os-level page caching* (enabled by
  default in FFCV) to ensure that concurrent training executions properly
  exploit caching.
- If your dataset does not fit in memory, use *process-level page caching*,
  (enabled by setting ``os_cache=False`` when constructing the
  :class:`ffcv.loader.Loader`) to avoid caching the entire dataset at once.
- Especially when using process-level caching, consider using the *quasi-random
  data sampler*, enabled using the ``order=OrderOption.QUASI_RANDOM`` argument to
  the :class:`~ffcv.loader.Loader` constructor. Quasi-random sampling tries to
  imitate random sampling while minimizing the underlying number of disk reads.
  (Again, note that ``QUASI_RANDOM`` is not yet supported for distributed training.)
- Another option for computer vision datasets is *storing resized images*: many
  datasets have gigantic images that end up being resized and cropped anyways in
  the data augmentation pipeline. You can avoid paying the cost of loading these
  giant images by writing them to an appropriate side length in the first place
  with :class:`ffcv.writer.DatasetWriter` (see the :ref:`Working with Image Data in FFCV` guide)
- Similarly, you can store images in JPEG format to save both disk space and
  reading time, and lower serialized JPEG quality to decrease storage sizes.

CPU bottlenecks
---------------
All CPUs at 100% and you're still not hitting maximal GPU usage? Consider the
following:

- Use premade, **JIT-compiled augmentations** from :mod:`ffcv.transforms`: these
  augmentations use pre-allocated pinned memory, and are fused together and
  compiled to machine code at runtime, making them a much faster alternative to
  standard data augmentation functions.
- **Make your own** JIT-compiled augmentations: If you don't see your desired
  augmentation among the pre-implemented ones, implementing your own efficient
  augmentation is simple and only requires implementing a single Python
  function. See any of the existing augmentations for an example, or read the
  `Customization guide <#>`_ (coming soon!) for a tutorial.
- *Store (some) raw pixel data* (<code>cv</code>): FFCV allows you to smoothly
  trade off I/O workload and compute workload (raw pixels require no JPEG decoding) by
  randomly storing a specified fraction of the dataset as raw pixel data instead
  of JPEG.

GPU bottlenecks
---------------
Even if you're not bottlenecked by data loading, FFCV can still help you
accelerate your system: 

- **Asynchronous CPU-GPU data transfer**: we always asynchronously transfer
  data, and also include tools for ensuring unblocked GPU execution.
- **Train multiple models on the same GPU**: Fully
  asynchronous thrad-based dataloading means that unlike for other data loading
  systems, different training processes using FFCV running on the same GPU won't
  block each other.
- **Offload compute to the CPU**: because FFCV offer extremely fast JIT-compiled
  data transformations, it's often helpful to move parts of the data pipeline (e.g.,
  input normalization or image augmentation) to CPU; FFCV will handle compilation
  and parallezation of these functions so that the CPU-induced slowdown isn't too
  great, and the freed-up GPU time can be used for more GPU-intensive tasks (e.g.,
  matrix multiplication).

.. note:: 

    This list is limited to what ffcv offers in data loading; check out
    guides like `the PyTorch performance guide
    <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_ for more
    model-based ways to speed up training. 