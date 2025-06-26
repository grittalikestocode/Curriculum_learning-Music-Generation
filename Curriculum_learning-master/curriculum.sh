#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40s:3

# Load required modules
#module load Python/3.11.5-GCCcore-13.2.0  # Adjust if needed
#module load CUDA/11.6.0      


# python traincl_finale.py checkpoints/cl_experiment2.pt models/cl_experiment2.pt 400
# python traincl_finale.py checkpoints/cl/cl_experiment_15.pt models/cl_experiment15.pt 1000
# python traincl_finale.py checkpoints/cl/cl_experiment1_epoch_150.pt models/cl_experiment15.pt 200 --load-checkpoint 
# python traincl_finale_80.py checkpoints/cl/cl_experiment15_80_new.pt models/cl_experiment15_80_new.pt 300 

# python traincl_finale_80.py checkpoints/cl/cl_experiment15_epoch_140.pt models/cl_experiment15_80_new.pt 300 --load-checkpoint 
# python traincl_finale_20.py checkpoints/cl60/cl_experiment15_epoch_120.pt models/cl_experiment15_60_new.pt 300  --load-checkpoint 

python traincl_learning.py checkpoints/cl_lr/cl_experiment15_epoch_120.pt models/cl_experiment15_60_lr.pt 300 --load-checkpoint