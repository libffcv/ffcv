<p align = 'center'>
<em>ImageNet is the new CIFAR: train models at <a href="#imagenet">1/10th the cost*</a> with accelerated data loading!</em>
</p>
<img src='assets/logo.png' width='100%'/>
<p align = 'center'>
[<a href="#installation">install</a>]
[<a href="#quickstart">quickstart</a>]
[<a href="#docs">docs</a>]
[<a href="#customdatasets">walkthrough</a>]
[<a href="#intro">ImageNet</a>]
[<a href="#intro">CIFAR</a>]
</p>

`ffcv` dramatically increases data throughput in accelerated computing systems, offering:
 - Fast data loading (even in resource constrained environments)
 - Efficient (yet easy to understand/customize) training code for standard computer vision tasks

With `ffcv` you can:
- ...break the [MLPerf record*](TODO) for ImageNet training: TODO min on 8 AWS GPUs
- ...train an ImageNet model on one GPU in TODO minutes (XX$ on AWS)
- ...train a CIFAR-10 model on one GPU in TODO seconds (XX$ on AWS)
- ...train a $YOUR_DATASET model $REALLY_FAST (for $WAY_LESS)

## Install
Via [Anaconda](https://docs.anaconda.com/anaconda/install/index.html):
```
conda create -n ffcv python=3.9 pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c pytorch -c conda-forge
conda activate ffcv
pip install ffcv
``` 

## Quickstart
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_imagenet.py --config-file imagenet_configs/resnet_linear_18_30.yaml --training.distributed=1 --dist.world_size=8

## 
