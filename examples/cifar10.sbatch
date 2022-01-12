#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH -a 0-6
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/nfs/home/aiilyas/slurm/scratch/cifar-logs/B-%a-%j.log

epochs=(10 15 20 25 30 35 40)
lrs=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0) 

for i in {0..9}
do
    python train_cifar.py --params.num_epochs ${epochs[$SLURM_ARRAY_TASK_ID]} \
                          --params.lr ${lrs[i]} \
                          --params.label_smoothing 0.0
    python train_cifar.py --params.num_epochs ${epochs[$SLURM_ARRAY_TASK_ID]} \
                          --params.lr ${lrs[i]} \
                          --params.label_smoothing 0.1
    python train_cifar.py --params.num_epochs ${epochs[$SLURM_ARRAY_TASK_ID]} \
                          --params.lr ${lrs[i]} \
                          --params.label_smoothing 0.2
done