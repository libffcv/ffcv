<p align = 'center'>
<em><b>Fast Forward Computer Vision</b>: train models at <a href="#imagenet">1/10th the cost*</a> with accelerated data loading!</em>
</p>
<img src='assets/logo.svg' width='100%'/>
<p align = 'center'>
[<a href="https://ffcv.io">homepage</a>]
[<a href="#installation">install</a>]
[<a href="#overview">overview</a>]
[<a href="https://docs.ffcv.io">docs</a>]
[<a href="#imagenet">ImageNet</a>]
[<a href="https://join.slack.com/t/ffcv-workspace/shared_invite/zt-11olgvyfl-dfFerPxlm6WtmlgdMuw_2A">support slack</a>]
<br>
Maintainers:
<a href="https://twitter.com/gpoleclerc">Guillaume Leclerc</a>,
<a href="https://twitter.com/andrew_ilyas">Andrew Ilyas</a> and
<a href="https://twitter.com/logan_engstrom">Logan Engstrom</a>
</p>

`ffcv` dramatically increases data throughput in accelerated computing systems,
offering:
- <a href="#overview">Fast data loading</a> (even in resource constrained environments)
- Efficient (yet Easy To Understand/customize) <a href="">training code</a> for standard
   computer vision tasks

Install `ffcv` today and:
- ...train an ImageNet model on one GPU in TODO minutes (XX$ on AWS)
- ...train a CIFAR-10 model on one GPU in 36 seconds (XX$ on AWS)
- ...train a `$YOUR_DATASET` model `$REALLY_FAST` (for `$WAY_LESS`)

Compare our training and dataloading times to what you use now: 
TODO
TODO

## Install
With [Anaconda](https://docs.anaconda.com/anaconda/install/index.html):

```
conda install ffcv
``` 

## Citation
If you find FFCV useful in your work, please cite it as:
```
@misc{leclerc2022ffcv,
    author = {Guillaume Leclerc and Andrew Ilyas and Logan Engstrom and Sung Min Park and Hadi Salman and Aleksander Madry},
    title = {ffcv},
    year = {2021},
    howpublished = {\url{https://github.com/MadryLab/ffcv/}},
    note = {commit xxxxxxx}
}
```

## In this document
- <a href="#overview"><b>Overview</b></a>: High level introduction to `ffcv`
- <a href="#features"><b>Features</b></a>Mapping of data loading problems to our solutions.
- <a href="TODO"><b>Results on ImageNet and CIFAR</b></a>: Results, code, and training configs for ImageNet

## Overview
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
We provide a self-contained script for training ImageNet <it>fast</it>. Below we
plot the training time versus
accuracy frontier for 1-GPU ResNet-18 and 8-GPU ResNet-50 alongside
a few baselines:

[resnet18 plot] [resnet50 plot]

**Train your own ImageNet models!** <a href="https://github.com/MadryLab/ffcv/tree/new_ver/examples/imagenet">Use our training script and premade configurations</a> to train any model seen on the above graphs.

### CIFAR-10
We also include premade code for efficient CIFAR-10 training in the `examples/` directory; we obtain 93\% top1 accuracy in 36 seconds with one A100 GPU. You can find the training script and configuration <a href="TODO">here</a>.

## Custom Datasets Quickstart
<!-- Accelerating data loading with `ffcv` requires two steps: dataset serialization
into the `ffcv` format, and then deploying the `ffcv` data loader at train-time. -->
Accelerating data loading with `ffcv` requires two steps: dataset preprocessing into `ffcv` format,
and then deploying the `ffcv` data loader at train-time. To help you tune the
options for each step, follow the guide below for two standard cases:

<p><b>Dataset fits in memory.</b> Either your dataset is small or DARPA awarded
your advisor 1 TB of RAM. Here, data
reading will likely not bottleneck training, so you should focus on reducing CPU
and GPU bottlenecks: 

- Loading (`ffcv.loader.Loader`) options: Always set `os_cache=True` to cache the entire
dataset.
- Writing (`DatasetWriter`) options: write examples such that loading is not CPU
bound. See a full list [below](TODO); some options here are downscaling stored images,
storing raw pixel values in place of JPEGs, or storing lower quality JPEGs to speed up decoding.
</p>
<p><b>Dataset does not fit in memory.</b> Many datasets will not fit in memory;
you should work towards reducing disk-read bottlenecks (or CPU bottlenecks if you can't decode JPEGs fast enough):

- Loading (`ffcv.loader.Loader`) options: Always set `os_cache=False` and if you want random ordering `order=OptionOrder.QUASI_RANDOM` (in place of `OptionOrder.RANDOM`).
- Writing (`DatasetWriter`) options: write examples such that loading is not CPU
or disk bound; see a full list of strategies [below](TODO). For example,
store lower quality or downsized JPEGs.
</p>

<p><b>General best practices.</b> For most <code>ffcv</code> applications:

- Replace data augmentations with `ffcv` <a href="TODO">built-in equivalents</a> when possible.
- <a href="TODO">Port your data augmentations</a> over to `ffcv` via <a href="TODO">Numba</a> if you have the time; `ffcv` does support slower, non-numba augmentations as well.
</p>

## Features
<img src='assets/clippy.png' width='100%'/>

Computer vision or not, name your performance bottleneck, and FFCV can help! See our  
<a href="https://docs.ffcv.io/performance_guide.html">performance guide</a> for a 
more detailed look.
(`cv` denotes computer-vision specific features)

<p><b>Disk-read bottlenecks.</b> What if your GPUs sit idle from low disk throughput?
Maybe you're reading from a networked drive, maybe you have too many GPUs;
either way, try these features:
<ul>
<li><b><a href="TODO">Use process-level page caching</a></b>: TODO</li>
<li><b><a href="TODO">Use os-level page caching</a></b>: TODO Assuming your <code>ffcv</code> dataset fits in memory, use os-level page caching to ensure that concurrent training executions properly exploit caching.</li>
<li><b><a href="TODO">Use quasi-random data sampling</a></b>: TODO (NOTE DOES NOT WORK WITH DISTRIBUTED)</li>
<li><b><a href="TODO">Store resized images</a></b> (<code>cv</code>): Many datasets have gigantic images even though most pipelines crop and resize to smaller edge lengths before training.</li>
<li><b><a href="TODO">Store JPEGs</a></b> (<code>cv</code>): Store images as space-efficient JPEGs.</li>
<li><b><a href="TODO">Store lower quality JPEGs</a></b> (<code>cv</code>): Lower serialized JPEG quality to decrease storage sizes.</li>
</ul>
</p>

<p><b>CPU bottlenecks.</b> All CPUs at 100% and you're still not hitting maximal
GPU usage? Consider the following:
<ul>
<li><b><a href="TODO">Use premade, JIT-compiled augmentations</a></b>: TODO use our premade chunguses instead of the standard chunguses.</li>
<li><b><a href="TODO">Make your own JIT-compiled augmentations</a></b>: Compile your optimizations into TODO </li>
<li><b><a href="TODO">Fuse together redundant operations</a></b>: TODO </li>
<li><b><a href="TODO">Store resized images</a></b> (<code>cv</code>): Smaller images require less compute to decode.</li>
<li><b><a href="TODO">Store lower quality JPEGs</a></b> (<code>cv</code>): Lower serialized JPEG quality to decrease CPU cycles spent decoding.</li>
<li><b><a href="TODO">Store a fraction of images as raw pixel data</a></b> (<code>cv</code>): Trade off storage and compute workload (raw pixels require no JPEG decoding) by randomly storing a specified fraction of the dataset as raw pixel data.</li>
</ul>
</p>

<p><b>GPU bottlenecks (any data).</b> Even if you're not bottlenecked by data
loading, <code>ffcv</code> can still accelerate your system:
<ul>
<li><b><a href="TODO">Asynchronous CPU-GPU data transfer</a></b>: While we always asynchronously transfer data, we also include tools for ensuring unblocked GPU execution.</li>
<li><b><a href="TODO">Train multiple models on the same GPU</a></b>: Fully
asynchronous dataloading means that different training processes won't block eachother.</li>
<li><b><a href="TODO">Offload compute to the CPU</a></b>: offload compute, like <a href="TODO">normalization</a> or <a href="">other augmentations</a>, onto the CPU.</li>
</ul>
This list is limited to what <code>ffcv</code> offers in data loading; check out
guides like <a href="https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html">the PyTorch performance guide</a> for more ways to speed
up training. 
