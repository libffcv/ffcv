#!/bin/bash

set -u -e

LR="$1"
WD="$2"
EPOCHS="$3"
OUT_DIR="$4"

CUDA_VISIBLE_DEVICES=8 python train_cifar.py \
    --data.train_dataset ~/datasets/cifar_betons/cifar_train.beton \
    --data.val_dataset ~/datasets/cifar_betons/cifar_val.beton \
    --model.arch resnet9 --training.epochs $EPOCHS --training.weight_decay $WD \
    --training.lr $LR --logging.folder $OUT_DIR --validation.lr_tta \
    --training.label_smoothing 0.05