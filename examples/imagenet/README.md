# `ffcv` ImageNet Training

## Models and Configurations
### Results

<img src="../../docs/_static/perf_scatterplot.svg" width='830px'/>

See benchmark setup here: [https://docs.ffcv.io/benchmarks.html](https://docs.ffcv.io/benchmarks.html).

### Configurations
The configs corresponding to the above results are:

| Link to Config                                                                                                                         |   top_1 |   top_5 |   # Epochs |   Time (mins) | Architecture   | Setup    |
|:---------------------------------------------------------------------------------------------------------------------------------------|--------:|--------:|-----------:|--------------:|:---------------|:---------|
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn50_configs/23cf4c2e-2111-472c-8b9d-e47325c51253.yaml'>Link</a> | 0.780 | 0.941  |         88 |       69.9 | ResNet-50      | 8 x A100 |
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn50_configs/a2e443e4-0a42-49b5-a1f4-18f19689b2b1.yaml'>Link</a> | 0.773 | 0.937 |         56 |       44.6 | ResNet-50      | 8 x A100 |
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn50_configs/70a2ecb0-1719-40bd-b1eb-cb0c37b8fc89.yaml'>Link</a> | 0.763 | 0.932 |         40 |       32.2 | ResNet-50      | 8 x A100 |
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn50_configs/2eca7655-82df-4e4a-b51b-950b31c2bebe.yaml'>Link</a> | 0.754 | 0.927 |         32 |       25.9 | ResNet-50      | 8 x A100 |
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn50_configs/c01efd78-d20b-4064-a9a8-aa0274430e60.yaml'>Link</a> | 0.746 | 0.921 |         24 |       19.6  | ResNet-50      | 8 x A100 |
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn50_configs/2018a9be-f0ec-453c-9a00-bfd68dc94d58.yaml'>Link</a> | 0.724 | 0.908 |         16 |       13.4 | ResNet-50      | 8 x A100 |
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn18_configs/67117b5b-ab11-45c2-87a6-9999a66a3f37.yaml'>Link</a> | 0.715 | 0.903   |         88 |      189.7  | ResNet-18      | 1 x A100 |
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn18_configs/1a584c27-fdd0-483b-b358-dd9895e510da.yaml'>Link</a> | 0.707  | 0.899 |         56 |      117.9   | ResNet-18      | 1 x A100 |
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn18_configs/8249f454-c718-4a38-8dae-04effa3a43de.yaml'>Link</a> | 0.698 | 0.894 |         40 |       85.4 | ResNet-18      | 1 x A100 |
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn18_configs/baefac94-ed5c-4b0b-af01-ebead347ca46.yaml'>Link</a> | 0.690 | 0.889 |         32 |       68.4   | ResNet-18      | 1 x A100 |
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn18_configs/05b0053a-c036-4992-9c4e-f9ea662abfc5.yaml'>Link</a> | 0.679  | 0.881 |         24 |       51.2 | ResNet-18      | 1 x A100 |
| <a href='https://github.com/MadryLab/ffcv/blob/main/examples/imagenet/rn18_configs/08508b1c-ae7f-4022-9090-06a520f998c8.yaml'>Link</a> | 0.655 | 0.868 |         16 |       34.8 | ResNet-18      | 1 x A100 |

## Training Models

First, generate an ImageNet dataset; make the dataset used for the results above with the following command (`IMAGENET_DIR` should point to a PyTorch style [ImageNet dataset](https://github.com/MadryLab/pytorch-imagenet-dataset):

```bash
# Required environmental variables for the script:
export IMAGENET_DIR=/path/to/pytorch/format/imagenet/directory/
export WRITE_DIR=/your/path/here/

# Starting in the root of the Git repo:
cd examples;

# Serialize images with:
# - 400px side length maximum
# - 10% JPEG encoded, 90% raw pixel values
# - quality=90 JPEGs
./write_dataset.sh 400 0.10 90
```
Then, choose a configuration from the [configuration table](#configurations). With the config file path in hand, train as follows:
```bash
# 1 GPU training
export CUDA_VISIBLE_DEVICES=0

# Set the visible GPUs according to the `world_size` configuration parameter
# Modify `data.in_memory` and `data.num_workers` based on your machine
python train_imagenet.py --config-file path/to/config/file.yaml \
    --data.train_dataset=/path/to/train/dataset.ffcv \
    --data.val_dataset=/path/to/val/dataset.ffcv \
	--data.num_workers=6 --data.in_memory=1 
```
Adjust the configuration by either changing the passed YAML file or by specifying arguments via [fastargs](https://github.com/GuillaumeLeclerc/fastargs) (i.e. how the dataset paths were passed above).
## Training Details
<p><b>System setup.</b> We trained on p4.24xlarge ec2 instances and on our own cluster machines (9 A100s / 504GB RAM / 48 cores).
</p>

<p><b>Algorithmic details.</b> We use a standard ImageNet training pipeline (à la the PyTorch ImageNet example) with only the following differences/highlights:

- SGD optimizer with momentum
- Test-time augmentation over left/right flips
- Validation set sizing according to ["Fixing the train-test resolution discrepancy"](https://arxiv.org/abs/1906.06423) 
- Progressive resizing from 160px to 196px
- Label smoothing
- Cyclic learning rate schedule
</p>

Refer to the code and configuration files for a more exact specification.
To obtain configurations we first gridded for hyperparameters at a 30 epoch schedule. Fixing these parameters, we then varied only the number of epochs (stretching the learning rate schedule across the number of epochs as motivated by [Budgeted Training](https://arxiv.org/abs/1905.04753)) and plotted the results above.

## FAQ
### How do I choose my dataset parameters?
If you want to reproduce our numbers you will need to make a dataset that can fully saturate your GPUs when loaded; 
we recommend making your dataset in-memory with the following (sweeping) guidelines depending on your system:

- \>500 GB RAM, >46 cores: Run `./write_dataset.sh 400 0.10 90`
- 300-500 GB RAM, >24 cores: Run `./write_dataset.sh 350 0.10 90`

These may not work depending on your system. Refer to [the performance guide](https://docs.ffcv.io/performance_guide.html) for guidelines that you can apply more specifically; we strongly recommend generating a dataset (a) small enough to fully fit in memory and (b) fast enough to decode that your GPU is saturated. 

### What if I can't fit my dataset in memory?
See this [guide here](https://docs.ffcv.io/parameter_tuning.html#scenario-large-scale-datasets).

### Other questions
Please open up a [GitHub discussion](https://github.com/MadryLab/ffcv/discussions) for non-bug related questions; if you find a bug please report it on [GitHub issues](https://github.com/MadryLab/ffcv/issues).