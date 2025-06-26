#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L4:2

# Load required modules
#module load Python/3.11.5-GCCcore-13.2.0  # Adjust if needed
#module load CUDA/11.6.0      


python scatter.py 