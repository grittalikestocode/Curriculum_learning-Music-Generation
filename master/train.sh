#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40s:3
# Load required modules
#module load Python/3.11.5-GCCcore-13.2.0  # Adjust if needed
#module load CUDA/11.6.0      

# python train.py large_data_15.pt checkpoints/experiment_new.pt models/experiment_new.pt 300
# python train.py processed_data.pt checkpoints/test.pt models/test.pt 1
# python train.py large_data_15.pt checkpoints/experiment_large_epoch_10_epoch_20_epoch_30_epoch_40_epoch_50_epoch_60_epoch_70_epoch_80_epoch_90_epoch_100_epoch_110_epoch_120_epoch_130_epoch_140_epoch_150_epoch_160_epoch_170_epoch_180_epoch_190_epoch_200_epoch_210_epoch_220_epoch_230_epoch_240.pt models/experiment_new.pt 60 --load-checkpoint
python train.py large_data_15.pt checkpoints/experiment_large.pt models/experiment_new.pt 100 