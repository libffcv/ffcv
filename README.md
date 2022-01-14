<p align = 'center'>
<em><b>Fast Forward Computer Vision</b>: train models at a fraction of the cost with accelerated data loading!</em>
</p>
<p align='center'>
[<a href="https://ffcv.io">homepage</a>]
[<a href="https://docs.ffcv.io">docs</a>]
[<a href="https://join.slack.com/t/ffcv-workspace/shared_invite/zt-11olgvyfl-dfFerPxlm6WtmlgdMuw_2A">support slack</a>]
</p>
<img src='assets/logo.svg' width='100%'/>
<p align = 'center'>
<!-- <br /> -->
[<a href="#install-with-anaconda">install</a>]
[<a href="#quickstart">quickstart</a>]
[<a href="#prepackaged-computer-vision-benchmarks">results</a>]
[<a href="#features">features</a>]
<br>
Maintainers:
<a href="https://twitter.com/gpoleclerc">Guillaume Leclerc</a>,
<a href="https://twitter.com/andrew_ilyas">Andrew Ilyas</a> and
<a href="https://twitter.com/logan_engstrom">Logan Engstrom</a>
</p>

`ffcv` dramatically increases data throughput in ML training.
In this repo, you will find:
- Our library for <a href="#quickstart">fast data loading and processing</a>
  (even in resource constrained environments) 
- Efficient, simple, easy-to-understand, customizable training code for standard
   vision tasks

See the [Features](#features) section below for a glance at what FFCV can do! Or
[install `ffcv`](#install-with-anaconda) today and:

- ...[train an ImageNet model]() on one GPU in 30 minutes (XX$ on AWS)
- ...[train a CIFAR-10 model]() on one GPU in 36 seconds (XX$ on AWS)
- ...train a `$YOUR_DATASET` model `$REALLY_FAST` (for `$WAY_LESS`)

Compare our training and dataloading times to what you might use now: 

<img src="assets/headline.svg"/>

Holding constant the same training routine and optimizing only the dataloading
and data transfer routines with `ffcv`, we enable significantly faster training:

<img src="docs/_static/perf_scatterplot.svg"/>

See [here](https://docs.ffcv.io/benchmarks.html) for further benchmark details.

## Install with Anaconda
```
conda install ffcv -c pytorch -c conda-forge -c ffcv
``` 

## Citation
If you use FFCV, please cite it as:

```
@misc{leclerc2022ffcv,
    author = {Guillaume Leclerc and Andrew Ilyas and Logan Engstrom and Sung Min Park and Hadi Salman and Aleksander Madry},
    title = {ffcv},
    year = {2021},
    howpublished = {\url{https://github.com/MadryLab/ffcv/}},
    note = {commit xxxxxxx}
}
```

## Quickstart
Accelerate <a href="#features">*any*</a> learning system with `ffcv`.
First,
convert your dataset into `ffcv` format (`ffcv` converts both indexed PyTorch datasets and
<a href="https://github.com/webdataset/webdataset">WebDatasets</a>):
```python
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField, NDArrayField
import numpy as np

# Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs
my_dataset = make_my_dataset() 
write_path = '/output/path/for/converted/ds.ffcv'

# Pass a type for each data field
writer = DatasetWriter(write_path, {
    # Tune options to optimize dataset size, throughput at train-time 
    'image': RGBImageField({
        max_resolution=256,
        jpeg_quality=jpeg_quality
    }),
    'label': IntField()
})

# Write dataset
writer.from_indexed_dataset(ds)
```
Then replace your old loader with the `ffcv` loader at train time (in PyTorch,
no other changes required!):
```python
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Cutout
from ffcv.fields.rgb_image import RandomResizedCropRGBImageDecoder

# Random resized crop
decoder = RandomResizedCropRGBImageDecoder((224, 224))

# Data decoding and augmentation
image_pipeline = [decoder, Cutout(), ToTensor(), ToTorchImage(), ToDevice(0)]
label_pipeline = [IntDecoder(), ToTensor(), ToDevice(0)]

# Pipeline for each data field
pipelines = {
    'image': image_pipeline,
    'label': label_pipeline
}

# Replaces PyTorch data loader (`torch.utils.data.Dataloader`)
loader = Loader(train_path, batch_size=bs, num_workers=num_workers,
                order=OrderOption.RANDOM, pipelines=pipelines)

# rest of training / validation proceeds identically
for epoch in range(epochs):
    ...
```
[See here](TODO) for a more detailed guide to deploying `ffcv` for your dataset.

## Prepackaged Computer Vision Benchmarks
From gridding to benchmarking to fast research iteration, there are many reasons
to want faster model training. Below we present premade codebases for training 
on ImageNet and CIFAR, including both (a) extensible codebases and (b)
numerous premade training configurations.

### ImageNet
We provide a self-contained script for training ImageNet <it>fast</it>. 
[Above](#plots) we plot the training time versus
accuracy frontier, and the dataloading speeds, for 1-GPU ResNet-18 and 8-GPU
ResNet-50 alongside a few baselines.

**Train your own ImageNet models!** You can <a href="https://github.com/MadryLab/ffcv/tree/new_ver/examples/imagenet">use our training script and premade configurations</a> to train any model seen on the above graphs.

### CIFAR-10
We also include premade code for efficient training on CIFAR-10 in the `examples/`
directory, obtaining 93\% top1 accuracy in 36 seconds on a single A100 GPU
(without optimizations such as MixUp, Ghost BatchNorm, etc. which have the
potential to raise the accuracy even further). You can find the training script
<a href="https://github.com/MadryLab/ffcv/tree/new_ver/examples/cifar">here</a>. 

## Features
<img src='docs/_static/clippy-transparent-2.png' width='100%'/>

Computer vision or not, FFCV can help make training faster in a variety of
resource-constrained settings!
Our <a href="https://docs.ffcv.io/performance_guide.html">performance guide</a>
has a more detailed account of the ways in which FFCV can adapt to different
performance bottlenecks.


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

  See our [Getting started](https://docs.ffcv.io/basics.html) guide, 
  [Example walkthroughs](https://docs.ffcv.io/examples.html), and 
  [Code examples](https://github.com/MadryLab/ffcv/tree/new_ver/examples) 
  to see how easy it is to get started!
- **Fast data processing without the pain**: FFCV automatically handles data
  reading, pre-fetching, caching, and transfer between devices in an extremely
  efficiently way, so that users don't have to think about it.
- **Automatically fused-and-compiled data processing**: By either using
  [pre-written](https://docs.ffcv.io/api/transforms.html) FFCV transformations 
  or 
  [easily writing custom ones](https://docs.ffcv.io/ffcv_examples/custom_transforms.html), 
  users can
  take advantage of FFCV's compilation and pipelining abilities, which will
  automatically fuse and compile simple Python augmentations to machine code
  using [Numba](https://numba.org), and schedule them asynchronously to avoid
  loading delays.
- **Load data fast from RAM, SSD, or networked disk**: FFCV exposes
  user-friendly options that can be adjusted based on the resources
  available. For example, if a dataset fits into memory, FFCV can cache it
  at the OS level and ensure that multiple concurrent processes all get fast
  data access. Otherwise, FFCV can use fast process-level caching and will
  optimize data loading to minimize the underlying number of disk reads. See
  [The Bottleneck Doctor](https://docs.ffcv.io/bottleneck_doctor.html) 
  guide for more information.
- **Training multiple models per GPU**: Thanks to fully asynchronous
  thread-based data loading, you can now interleave training multiple models on
  the same GPU efficiently, without any data-loading overhead. See 
  [this guide](https://docs.ffcv.io/parameter_tuning.html) for more info.  
- **Dedicated tools for image handling**: All the features above work are
  equally applicable to all sorts of machine learning models, but FFCV also
  offers some vision-specific features, such as fast JPEG encoding and decoding,
  storing datasets as mixtures of raw and compressed images to trade off I/O
  overhead and compute overhead, etc. See the 
  [Working with images](https://docs.ffcv.io/working_with_images.html) guide for
  more information.

# Contributors

- Guillaume Leclerc
- Logan Engstrom
- Andrew Ilyas
- Sam Park
- Hadi Salman
