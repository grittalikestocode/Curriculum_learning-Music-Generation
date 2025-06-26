#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L4:1

# Load required modules
#module load Python/3.11.5-GCCcore-13.2.0  # Adjust if needed
#module load CUDA/11.6.0      


# python curriculumsort.py processed_data.pt models/save_path_final.pt sorted1cl1.pt
# python computedifficulty1.py checkpoints/ckpt_path_final.pt sortedsequence/sorted1cl2.pt
# python computed_difficulty.py checkpoints/ckpt_path_final.pt val_ds.pt sortedsequence/sorted1cl6.pt

python computed_difficulty.py checkpoints/experiment_large_epoch_300.pt train_ds_15.pt sortedsequence/cl/experiment15.pt