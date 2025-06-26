#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L4:1

# Load required modules
#module load Python/3.11.5-GCCcore-13.2.0  # Adjust if needed
#module load CUDA/11.6.0      


# python generate.py models/experiment_new.pt generate_music/baseline.mid -v
python generate.py baseline1.pt generate_music/baseline1.mid -v