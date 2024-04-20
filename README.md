# Fast Forward Computer Vision for Pretraining

<p align = 'center'>
<!-- <br /> -->
[<a href="#install-with-anaconda">install</a>]
[<a href="#features">new features</a>]
[<a href="https://docs.ffcv.io">docs</a>]
[<a href="https://arxiv.org/abs/2306.12517">paper</a>]
</p>

This library is derived from [FFCV](https://github.com/libffcv/ffcv) to optimize the memory usage and accelerate data loading. 

## Installation
### Running Environment
```
conda create -y -n ffcv "python>=3.9" cupy pkg-config "libjpeg-turbo>=3.0.0" opencv numba -c conda-forge
conda activate ffcv
conda install pytorch-cuda=11.3 torchvision  -c pytorch -c nvidia
pip install .
```

## Prepackaged Computer Vision Benchmarks
From gridding to benchmarking to fast research iteration, there are many reasons
to want faster model training. Below we present premade codebases for training
on ImageNet and CIFAR, including both (a) extensible codebases and (b)
numerous premade training configurations.

### ImageNet
We provide a self-contained script for training ImageNet <it>fast</it>.
Above we plot the training time versus
accuracy frontier, and the dataloading speeds, for 1-GPU ResNet-18 and 8-GPU
ResNet-50 alongside a few baselines.

TODO:

| Link to Config                                                                                                                         |   top_1 |   top_5 |   # Epochs |   Time (mins) | Architecture   | Setup    |
|:---------------------------------------------------------------------------------------------------------------------------------------|--------:|--------:|-----------:|--------------:|:---------------|:---------|
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn50_configs/rn50_88_epochs.yaml'>Link</a> | 0.784 | 0.941  |         88 |       77.2 | ResNet-50      | 8 x A100 |
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn50_configs/rn50_56_epochs.yaml'>Link</a> | 0.780 | 0.937 |         56 |       49.4 | ResNet-50      | 8 x A100 |
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn50_configs/rn50_40_epochs.yaml'>Link</a> | 0.772 | 0.932 |         40 |       35.6 | ResNet-50      | 8 x A100 |
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn50_configs/rn50_32_epochs.yaml'>Link</a> | 0.766 | 0.927 |         32 |       28.7 | ResNet-50      | 8 x A100 |
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn50_configs/rn50_24_epochs.yaml'>Link</a> | 0.756 | 0.921 |         24 |       21.7  | ResNet-50      | 8 x A100 |
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn50_configs/rn50_16_epochs.yaml'>Link</a> | 0.738 | 0.908 |         16 |       14.9 | ResNet-50      | 8 x A100 |
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn18_configs/rn18_88_epochs.yaml'>Link</a> | 0.724 | 0.903   |         88 |      187.3  | ResNet-18      | 1 x A100 |
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn18_configs/rn18_56_epochs.yaml'>Link</a> | 0.713  | 0.899 |         56 |      119.4   | ResNet-18      | 1 x A100 |
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn18_configs/rn18_40_epochs.yaml'>Link</a> | 0.706 | 0.894 |         40 |       85.5 | ResNet-18      | 1 x A100 |
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn18_configs/rn18_32_epochs.yaml'>Link</a> | 0.700 | 0.889 |         32 |       68.9   | ResNet-18      | 1 x A100 |
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn18_configs/rn18_24_epochs.yaml'>Link</a> | 0.688  | 0.881 |         24 |       51.6 | ResNet-18      | 1 x A100 |
| <a href='https://github.com/libffcv/ffcv-imagenet/tree/main/rn18_configs/rn18_16_epochs.yaml'>Link</a> | 0.669 | 0.868 |         16 |       35.0 | ResNet-18      | 1 x A100 |

**Train your own ImageNet models!** You can <a href="https://github.com/libffcv/imagenet-example/tree/main">use our training script and premade configurations</a> to train any model seen on the above graphs.

### CIFAR-10
We also include premade code for efficient training on CIFAR-10 in the `examples/`
directory, obtaining 93\% top1 accuracy in 36 seconds on a single A100 GPU
(without optimizations such as MixUp, Ghost BatchNorm, etc. which have the
potential to raise the accuracy even further). You can find the training script
<a href="https://github.com/libffcv/ffcv/tree/main/examples/cifar">here</a>.

## Features

Compared to the original FFCV, this library has the following new features:

- **crop decode**: RandomCrop and CenterCrop are now implemented to decode the crop region, which can save memory and accelerate decoding.

- **cache strategy**: There is a potential issue that the OS cache will be swapped out. We use `FFCV_DEFAULT_CACHE_PROCESS` to control the cache process. The choices for the cache process are:
  - `0`: os cache
  - `1`: process cache
  - `2`: Shared Memory 
  - `3`: redis: not implemented yet. May be used for large-scale datasets.
  
- **lossless compression**: PNG is supported for lossless compression. We use `RGBImageField(mode='png')` to enable the lossless compression.
