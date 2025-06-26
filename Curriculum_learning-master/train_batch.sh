#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40s:3

# Load required modules
#module load Python/3.11.5-GCCcore-13.2.0  # Adjust if needed
#module load CUDA/11.6.0      

python train_batch.py large_data_15.pt checkpoints/batch100/seed15.pt models/batch100_seed15.pt 100