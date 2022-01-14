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

<!-- In this repo, you will find:
- Our library for <a href="#quickstart">fast data loading and processing</a>
  (even in resource constrained environments)
- Efficient, simple, easy-to-understand, customizable training code for standard
   vision tasks -->

<!-- See the [Features](#features) section below for a glance at what FFCV can do! Or
[install `ffcv`](#install-with-anaconda) today and: -->

`ffcv` is a drop-in data loading system that dramatically increases data throughput in model training that enables...

- [Training an ImageNet model](https://github.com/MadryLab/ffcv/tree/main/examples/imagenet)
on one GPU in 35 minutes (98¢/model on AWS)
- [Training a CIFAR-10 model](https://docs.ffcv.io/ffcv_examples/cifar10.html)
on one GPU in 36 seconds (2¢/model on AWS)
- Training a `$YOUR_DATASET` model `$REALLY_FAST` (for `$WAY_LESS`)

<!-- Holding constant the same training routine and optimizing only the dataloading
and data transfer routines with `ffcv`, we enable significantly faster training: -->
Keep your training system the same, just replace the data loader. Look at these speeds!

<img src="assets/headline.svg"/>

`ffcv` also comes prepacked with fast, simple training code for
standard benchmarks: 

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
[See here](https://docs.ffcv.io/basics.html) for a more detailed guide to deploying `ffcv` for your dataset.

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

| Link to Config                                                                                                                         |   top_1 |   top_5 |   # Epochs |   Time (mins) | Architecture   | Setup    |
|:---------------------------------------------------------------------------------------------------------------------------------------|--------:|--------:|-----------:|--------------:|:---------------|:---------|
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn50_configs/7dc8b207-dd8f-405e-bbc1-61bf387e5ba5.yaml'>Link</a> | 0.77996 | 0.9406  |         88 |       69.9297 | ResNet-50      | 8 x A100 |
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn50_configs/33990e2f-96b1-4b34-9ea6-a90be740bb7b.yaml'>Link</a> | 0.77298 | 0.93742 |         56 |       44.5882 | ResNet-50      | 8 x A100 |
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn50_configs/9d44df9f-e73d-4730-84fa-170c9fcd98aa.yaml'>Link</a> | 0.76314 | 0.93198 |         40 |       32.2155 | ResNet-50      | 8 x A100 |
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn50_configs/d2325c91-210e-4212-b935-a04675bba779.yaml'>Link</a> | 0.75374 | 0.92746 |         32 |       25.8984 | ResNet-50      | 8 x A100 |
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn50_configs/5a3a96cf-db8b-4ba1-986e-bbc154f9ea3a.yaml'>Link</a> | 0.74618 | 0.92108 |         24 |       19.647  | ResNet-50      | 8 x A100 |
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn50_configs/641fa341-88f2-4382-9306-c3fe3fbad70f.yaml'>Link</a> | 0.72386 | 0.90848 |         16 |       13.4037 | ResNet-50      | 8 x A100 |
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn18_configs/663dd2ac-b127-4073-b824-6f19f0d15193.yaml'>Link</a> | 0.71456 | 0.903   |         88 |      189.736  | ResNet-18      | 1 x A100 |
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn18_configs/8f21c0e9-9d68-4c4b-899b-e32770503fff.yaml'>Link</a> | 0.7066  | 0.89854 |         56 |      117.88   | ResNet-18      | 1 x A100 |
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn18_configs/9cb3047b-ed52-4b3a-a3fb-7fe995063865.yaml'>Link</a> | 0.69848 | 0.89444 |         40 |       85.4398 | ResNet-18      | 1 x A100 |
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn18_configs/d683ce23-431e-4f7d-bfd2-6ee26882f4e4.yaml'>Link</a> | 0.69046 | 0.88852 |         32 |       68.39   | ResNet-18      | 1 x A100 |
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn18_configs/693bdc59-3dba-45b4-ae4c-8c274da87b95.yaml'>Link</a> | 0.6786  | 0.88094 |         24 |       51.2127 | ResNet-18      | 1 x A100 |
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn18_configs/2294bc2a-9f6d-420e-be91-6acff51f7f22.yaml'>Link</a> | 0.65478 | 0.86844 |         16 |       34.8248 | ResNet-18      | 1 x A100 |

**Train your own ImageNet models!** You can <a href="https://github.com/MadryLab/ffcv/tree/main/examples/imagenet">use our training script and premade configurations</a> to train any model seen on the above graphs.

### CIFAR-10
We also include premade code for efficient training on CIFAR-10 in the `examples/`
directory, obtaining 93\% top1 accuracy in 36 seconds on a single A100 GPU
(without optimizations such as MixUp, Ghost BatchNorm, etc. which have the
potential to raise the accuracy even further). You can find the training script
<a href="https://github.com/MadryLab/ffcv/tree/main/examples/cifar">here</a>.

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
  [Code examples](https://github.com/MadryLab/ffcv/tree/main/examples)
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
