#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L4:3
# Load required modules
#module load Python/3.11.5-GCCcore-13.2.0  # Adjust if needed
#module load CUDA/11.6.0     


python KLD.py --baseline "checkpoints/baseline.pt" --curriculum "checkpoints/cl/cl_experiment15_epoch_182_final.pt" --dataset "val_ds_15.pt" --save-dir "KLD_results_baseline_80%"

#cl60/cl_experiment15_epoch_162_final.pt