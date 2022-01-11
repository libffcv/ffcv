<p align = 'center'>
<em><b>Fast Forward Computer Vision</b>: train models at <a href="#imagenet">1/10th the cost*</a> with accelerated data loading!</em>
</p>
<img src='assets/logo.png' width='100%'/>
<p align = 'center'>
[<a href="#installation">install</a>]
[<a href="#overview">overview</a>]
[<a href="#docs">docs</a>]
[<a href="#intro">ImageNet</a>]
[<a href="#intro">CIFAR</a>]
[<a href="#intro">custom datasets</a>]
[<a href="#intro">FAQ</a>]
</p>

`ffcv` dramatically increases data throughput in accelerated computing systems,
offering:
 - Fast data loading (even in resource constrained environments)
 - Efficient (yet Easy To Understand/customize) training code for standard
   computer vision tasks

Install `ffcv` today and:
- ...break the <span>[MLPerf record*](TODO)</span> for ImageNet training: TODO min on 8 AWS GPUs
- ...train an ImageNet model on one GPU in TODO minutes (XX$ on AWS)
- ...train a CIFAR-10 model on one GPU in TODO seconds (XX$ on AWS)
- ...train a `$YOUR_DATASET` model `$REALLY_FAST` (for `$WAY_LESS`)

## Install
Via [Anaconda](https://docs.anaconda.com/anaconda/install/index.html):
```
conda create -n ffcv python=3.9 pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c pytorch -c conda-forge
conda activate ffcv
pip install ffcv
``` 

## Quick links
- <a href="TODO"><b>Overview</b></a>: High level introduction to `ffcv`
- <a href="TODO"><b>Quickstart</b></a>: Use `ffcv` to load data today
- <a href="TODO"><b>Feature Atlas</b></a>: How can `ffcv` help you? Mapping of data loading problems to our solutions.
- <a href="TODO"><b>ImageNet</b></a>: Results, code, and training configs for ImageNet
- <a href="TODO"><b>CIFAR</b></a>: Results, code, and training configs for CIFAR
- <a href="TODO"><b>Data loading benchmarks</b></a>:
- <a href="TODO"><b>Documentation</b></a>:
- <a href="TODO"><b>FAQ</b></a>:

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

## Feature Atlas
Why use `ffcv`? Computer vision or not, name your bottleneck, and we'll fix it! `cv` denotes computer-vision specific.
If you don't know how to identify your bottleneck consider reading <a href="TODO">our guide.</a>
<p><b>Disk-read bottlenecks.</b> What if your GPUs sit idle from low disk throughput?
Maybe you're reading from a networked drive, maybe you have too many GPUs;
either way, try these features:
<ul>
<!-- <li><a href="TODO">Store your dataset in memory</a>: TODO</li> -->
<li><b><a href="TODO">Use process-level page caching</a></b>: TODO</li>
<li><b><a href="TODO">Use os-level page caching</a></b>: TODO Assuming your <code>ffcv</code> dataset fits in memory, use os-level page caching to ensure that concurrent training executions properly exploit caching.</li>
<li><b><a href="TODO">Use quasi-random data sampling</a></b>: TODO</li>
<li><b><a href="TODO">Store resized images</a></b> (<code>cv</code>): Many datasets have gigantic images even though most pipelines crop and resize to smaller edge lengths before training.</li>
<li><b><a href="TODO">Store JPEGs</a></b> (<code>cv</code>): Store images as space-efficient JPEGs.</li>
<li><b><a href="TODO">Store lower quality JPEGs</a></b> (<code>cv</code>): Lower serialized JPEG quality to decrease storage sizes.</li>
</ul>
</p>

<p><b>CPU bottlenecks.</b> All CPUs at 100% and you're still not hitting maximal
GPU usage? Consider the following:
<ul>
<!-- <li><a href="TODO">Augment on GPU</a>: Offload CPU augmentation routines to the GPU.</li> -->
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
<!-- <li><a href="TODO">Augment on GPU</a>: Offload CPU augmentation routines to the GPU.</li> -->
<li><b><a href="TODO">Asynchronous CPU-GPU data transfer</a></b>: While we always asynchronously transfer data, we also include tools for ensuring unblocked GPU execution.</li>
<li><b><a href="TODO">Offload compute to the CPU</a></b>: offload compute, like <a href="TODO">normalization</a> or <a href="">other augmentations</a>, onto the CPU.</li>
<!-- <li>Optimized memory allocation: No hassle memory management.</li> -->
</ul>


## ImageNet

## CIFAR

## 


##  FAQ / Caveats
