#!/bin/bash
#SBATCH --job-name=explore_radio_data
#SBATCH --account=def-hinat
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=explore_radio_%j.out
#SBATCH --error=explore_radio_%j.err

# Load necessary modules
module load python/3.11 scipy-stack cuda/12.2 mpi4py opencv

# Activate your virtual environment
source ~/consistency_models_env/bin/activate

# Change to the consistency_models directory
cd /home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models:$PYTHONPATH"

echo "Starting radio data exploration..."
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"

# Run the exploration script
python explore_radio_data.py

echo "Exploration completed at: $(date)"
