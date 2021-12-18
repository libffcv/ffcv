#!/bin/bash

write_dataset () {
	python ../scripts/write_image_datasets.py \
		--cfg.dataset=imagenet \
		--cfg.split=${1} \
		--cfg.data_dir=/mnt/cfs/datasets/pytorch_imagenet \
		--cfg.write_path=/mnt/cfs/home/engstrom/store/ffcv/${1}_150.ffcv \
		--cfg.max_resolution=150
}

write_dataset train
write_dataset val