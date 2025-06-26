#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:L40s:2

# Load required modules
#module load Python/3.11.5-GCCcore-13.2.0  # Adjust if needed
#module load CUDA/11.6.0      


python validation.py