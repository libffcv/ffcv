# ImageNet with `ffcv`
We provide a self-contained script for training ImageNet efficiently: `train_imagenet.py`. To use it in research or on downstream applications we recommend copying the file and modifying it in place as needed (for example, to add features or additional logging). The script's accuracy vs training time graph is as follows for ResNet-18s (single A100 GPU) and ResNet-50s (8 A100 GPUs):

TODO

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
Then, choose a configuration from the [configuration table](TODO). With the config file path in hand, train as follows:
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

<p><b>Algorithmic details.</b> We use a standard ImageNet training pipeline (à la the PyTorch ImageNet example) with the following differences/highlights:

- SGD optimizer with momentum
- Test-time augmentation over left/right flips
- Validation set sizing according to ["Fixing the train-test resolution discrepancy"](https://arxiv.org/abs/1906.06423) 
- Progressive resizing from 160px to 224px
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

These may not work depending on your system. Refer to [TODO](TODO) for general guidelines that you can apply more specifically; we strongly recommend generating a dataset (a) small enough to fully fit in memory and (b) fast enough to decode that your GPU is saturated. 

### What if I can't fit my dataset in memory?
First look at the [guide here](todo); if you still can't succeed, use the flag `data.in_memory=0` to set the required settings for disk-limited training,  

### Other questions
Please open up a [GitHub discussion](https://github.com/MadryLab/ffcv/discussions) for non-bug related questions; if you find a bug please report it on [GitHub issues](https://github.com/MadryLab/ffcv/issues).