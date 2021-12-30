#!/bin/bash

set -a

LEARNING_RATE="$1"
END_RAMP="$2"
MIN_RES="$3"
MAX_RES="$4"
MOMENTUM="$5"
LABEL_SMOOTHING="$6"

set -e

BS=512
ARCH=resnet18

TRAIN_PATH=/mnt/nfs/datasets/imgnet_betons/imgnet_train_320px.beton
VAL_PATH=/mnt/nfs/datasets/imgnet_betons/imgnet_val_320px.beton

### FREE PARAMETERS
python train_imagenet.py --data.train_dataset $TRAIN_PATH \
    --data.val_dataset $VAL_PATH --data.gpu 0 --data.num_workers 11 \
    --logging.folder /tmp/ --model.arch $ARCH --training.batch_size $BS \
    --training.lr $LEARNING_RATE --training.momentum $MOMENTUM \
    --resolution.min_res $MIN_RES --resolution.max_res $MAX_RES \
    --resolution.end_ramp $END_RAMP --
