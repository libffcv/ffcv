ImageNet Benchmarks
====================

We benchmark our system using the `ImageNet <https://www.image-net.org>`_ dataset,
covering dataset size (storage), data loading,
and end-to-end training.
As we demonstrate below, FFCV significantly outperforms existing systems such as
`Pytorch DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_, `Webdataset <https://github.com/webdataset/webdataset>`_, and `DALI <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/>`_, while being much easier to use and extend.

Dataset sizes
--------------

In order to provide an idea of how the image encoding settings influence the resulting dataset, we generated multiple ImageNet datasets with various options. We present the results below. For more details about the image encoding options, please refer to :ref:`Working with Image Data in FFCV`.

We vary between three encoding options (JPEG, Mix (``proportion``), and RAW) and
four sizes (256px, 384px, 512px, 1600px).

.. image:: _static/dataset_sizes.svg
  :width: 60%
  :align: center
  :alt: Alternative text

|

.. list-table:: Dataset sizes
   :widths: 16 16 16 16 16 16
   :header-rows: 1

   * - Image Format
     - Quality
     - Size @ 256px
     - Size @ 384px
     - Size @ 512px
     - Size @ 1600px
   * - JPEG
     - 50
     - 9.23 GB
     - 16.14 GB
     - 26.35 GB
     - 30.75 GB
   * - JPEG
     - 90
     - 22.01 GB
     - 40.31 GB
     - 65.47 GB
     - 74.98 GB
   * - JPEG
     - 100
     - 57.00 GB
     - 110.21 GB
     - 176.65 GB
     - 198.53 GB
   * - Mix
     - 50
     - 49.59 GB
     - 102.29 GB
     - 173.92 GB
     - 221.76 GB
   * - Mix
     - 90
     - 58.36 GB
     - 124.74 GB
     - 202.04 GB
     - 251.66 GB
   * - Mix
     - 100
     - 84.91 GB
     - 176.43 GB
     - 285.67 GB
     - 350.72 GB
   * - RAW
     - N.A
     - 169.79 GB
     - 371.20 GB
     - 616.18 GB
     - 788.97 GB


Data loading
------------

Next, we measured the data loading performance of FFCV on some of the generated datasets from above when loaded from:

- RAM, simulating the case where the dataset is smaller than the amount of RAM available for caching.
- EBS (network attached drives on AWS), simulating the worst case scenario one would encounter on large datasets that are too big to be cached and even be stored on local storage.

We compare our results against existing data loading platforms:

- `Pytorch DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_: This is the default option that comes with the Pytorch library and uses individual JPEG files as the source.
- `Webdataset <https://github.com/webdataset/webdataset>`_: This loader requires pre-processed files aggregated in multiple big `.tar` archives.
- `DALI <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/>`_: Data loading pipeline developed by Nvidia. In this experiment we used the default file format which is the same as that of the Pytorch DataLoader.
The specific instantiation of DALI that we apply is the PyTorch ImageNet example DALI code found in the `NVIDIA DeepLearningExamples repository <https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/resnet50v1.5>`_.
We use the DGX-1 configuration and remove all the model optimization, benchmarking only the dataloader.


.. image:: _static/benchmarking_results.svg
  :width: 100%
  :align: center
  :alt: Alternative text


.. list-table:: Data loading benchmark results (ImageNet)
   :header-rows: 1

   * - Framework
     - Data Source
     - Resolution
     - Mode
     - All cores throughput (images/sec)
   * - FFCV
     - RAM
     - 512
     - JPEG 90%
     - 31278
   * - FFCV
     - RAM
     - 256
     - RAW
     - 172801
   * - FFCV
     - EBS
     - 512
     - RAW
     - 1956
   * - FFCV
     - EBS
     - 512
     - JPEG 90%
     - 16631
   * - FFCV
     - EBS
     - 256
     - RAW
     - 6870

.. note::
    The benchmarks were run on an AWS ``p3dn.24xlarge`` instance.

    For a fair comparison the baseline frameworks were evaluated on similarly resized datasets.

    The data loading pipeline consists of:

    - Loading the images
    - Random resized crop to 224x224 px
    - Random horizontal flip

End-to-end training
--------------------
Training ResNet-18s and ResNet-50s on ImageNet using code `here <https://github.com/libffcv/ffcv-imagenet/tree/main/>`_,
we plot the results below:

.. image:: _static/headline.svg
  :width: 90%
  :align: center
  :alt: Alternative text

|

For the same accuracy, we obtain much faster ImageNet training time than
the tested baselines. All testing was performed on a *p4d.24xlarge* AWS instance
with 8 A100s, and were given a training run before to warm up.
We tested two distinct benchmarks:

- ImageNet (Resnet-50 8xA100): Train a ResNet-50 on ImageNet with 8 A100s using data parallelism.
- ImageNet (Resnet-18 1xA100): Train a ResNet-18 on ImageNet with 1 A100. 

To make the benchmark realistic, we mimic standard cluster conditions by training 8 models at once, each on a separate GPU. Such training parallelism situations are also highly relevant for tasks like grid searching or finding confidence intervals for training results.

We detail the tested systems below:

- **FFCV**: We train using the code and system detailed  `in our repository <https://github.com/libffcv/ffcv-imagenet/tree/main/>`_.
- **PyTorch Example**: This is the popular ImageNet training code found
  `the PyTorch repository <https://github.com/pytorch/examples/blob/master/imagenet/main.py>`_.
  we measured the time to complete an epoch of training (after warmup) and then
  used that to extrapolate how long the implemented schedule would take. We took
  accuracies from
  `PyTorch model hub <https://pytorch.org/hub/pytorch_vision_resnet/>`_,
  assuming a 90 epoch schedule (a lower bound; the original ResNet paper used 120).
  We modified the PyTorch example to add half precision training (via PyTorch nativeAMP).
- **PyTorch Lightning**: Another popular training library, we used the example
  code from `the Lightning repository <https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/domain_templates/imagenet.py>`_,
  removed the import on line 46, and called the file with the DDP accelerator and
  half precision. We measured single epoch time (after warmup) and
  then, similar to the PyTorch example, assumed a 90 epoch schedule and correctness:
  that the resulting model would obtain the accuracy of a standard ResNet-50 trained
  on 90 epochs (i.e. the number listed in `PyTorch hub <https://pytorch.org/hub/pytorch_vision_resnet/>`_).
- **NVIDIA PyTorch**: NVIDIA's PyTorch ImageNet implementation, number and time lifted
  from the
  `website <https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/resnet50v1.5/README.md#results>`_.
- **TIMM A3**: The TIMM A3 ResNet-50 from
  `ResNet Strikes Back <https://arxiv.org/abs/2110.00476>`_.
  The paper originally used 4 V100s in training, so we assumed perfect scaling and
  lower bounded the training time by dividing the reported training time
  (15 hours) by 4 (V100s are at most
  `twice as slow <https://lambdalabs.com/blog/nvidia-a100-vs-v100-benchmarks/>`_
  as A100s and we used 8 GPUs instead of 4).
