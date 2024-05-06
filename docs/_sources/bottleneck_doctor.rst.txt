The Bottleneck Doctor
======================
.. image:: /_static/clippy-transparent-2.png
  :width: 100%

To summarize the scenarios from the :ref:`Tuning Guide`, we provide a map from a
type of *system bottleneck* to the FFCV options that will help get the most
performance out of your system:  

Disk-read bottlenecks 
---------------------
What if your GPUs sit idle from low disk or throughput?
Maybe you're reading from a networked drive, maybe you have too many GPUs;
either way, try:

- If your dataset fits in memory, use **OS-level page caching** (enabled by
  default in FFCV) to ensure that concurrent training executions properly
  exploit caching.
- If your dataset does not fit in memory, use **process-level page caching**,
  (enabled by setting ``os_cache=False`` when constructing the
  :class:`ffcv.loader.Loader`) to avoid caching the entire dataset at once.
- Especially when using process-level caching, consider using the **quasi-random
  data sampler**, enabled using the ``order=OrderOption.QUASI_RANDOM`` argument to
  the :class:`~ffcv.loader.Loader` constructor. Quasi-random sampling tries to
  imitate random sampling while minimizing the underlying number of disk reads.
  (Again, note that ``QUASI_RANDOM`` is not yet supported for distributed training.)
- Another option for computer vision datasets is **storing resized images**: many
  datasets have gigantic images that end up being resized and cropped anyways in
  the data augmentation pipeline. You can avoid paying the cost of loading these
  giant images by writing them to an appropriate side length in the first place
  with :class:`ffcv.writer.DatasetWriter` (see the :ref:`Working with Image Data in FFCV` guide)
- Similarly, you can **store images in JPEG format** to save both disk space and
  reading time, and lower serialized JPEG quality to decrease storage sizes.

CPU bottlenecks
---------------
All CPUs at 100% and you're still not hitting maximal GPU usage? Consider the
following:

- Use pre-made, **JIT-compiled augmentations** from :mod:`ffcv.transforms`: these
  augmentations use pre-allocated pinned memory, and are fused together and
  compiled to machine code at runtime, making them a much faster alternative to
  standard data augmentation functions.
- **Make your own** JIT-compiled augmentations: If you don't see your desired
  augmentation among the pre-implemented ones, implementing your own efficient
  augmentation is simple and only requires implementing a single Python
  function. See any of the existing augmentations for an example, or read the
  `Customization guide <#>`_ (coming soon!) for a tutorial.
- *Store (some) raw pixel data*: FFCV allows you to smoothly
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
  asynchronous thread-based data loading means that unlike for other data loading
  systems, different training processes using FFCV running on the same GPU won't
  block each other.
- **Offload compute to the CPU**: because FFCV offer extremely fast JIT-compiled
  data transformations, it's often helpful to move parts of the data pipeline (e.g.,
  input normalization or image augmentation) to CPU; FFCV will handle compilation
  and parallelization of these functions so that the CPU-induced slowdown isn't too
  much, and the freed-up GPU time can be used for more GPU-intensive tasks (e.g.,
  matrix multiplication).

.. note:: 

    This list is limited to what FFCV offers in data loading; check out
    guides like `the PyTorch performance guide
    <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_ for more
    model-based ways to speed up training. 
