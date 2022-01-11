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
- ...break the [MLPerf record*](TODO) for ImageNet training: TODO min on 8 AWS GPUs
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
- Overview: high level introduction to `ffcv`
- Features: how can `ffcv` help you? List of features.
- Quickstart: "Choose Your Own Adventure" guide to `ffcv` with your dataset and cluster
- ImageNet: results, code, and training configs for ImageNet
- CIFAR: results, code, and training configs for CIFAR
- Data loading benchmarks
- Documentation
- FAQ

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

## Features
Why use `ffcv`? Name your slowdown, and we'll fix it!
<p><b>Disk bottlenecks.</b> Reduce data transfer requirements by storing
<a href="">downsized images</a> or <a href="">adjusted quality JPEGs</a>.
Exploit 
</p>

</p>


<p><b>CPU bottlenecks (image data).</b> Reduce compute requirements by <a href="">storing
raw pixel values instead of JPEGs</a>. While this is expensive storage wise, you can
<a href="TODO">control the space/compute tradeoff</a> by only compressing a
(specified) fraction of the dataset. Finally, reduce decoding workload
by <a href="">adjusting the JPEG quality</a> or storing <a href="TODO">downsized
images</a>.</p>

<p><b>GPU bottlenecks (any data).</b> Even if you aren't bottlenecked by data
loading, <code>ffcv</code> can still accelerate your system with built-in
optimized memory allocation and asynchronous CPU-GPU data transfer.
We can also offload compute, like <a href="TODO">normalization</a>, onto the CPU.
</p>

## ImageNet

## CIFAR

## 


##  FAQ / Caveats
