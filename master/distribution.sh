#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40s:1

# Load required modules
#module load Python/3.11.5-GCCcore-13.2.0  # Adjust if needed
#module load CUDA/11.6.0      


# python curriculumsort.py processed_data.pt models/save_path_final.pt sorted1cl1.pt
# python distribution.py checkpoints/ckpt_path_final.pt sortedsequence/sorted1cl3.pt
python distribution.py models/final_model300 sortedsequence/sorted1cl3.pt