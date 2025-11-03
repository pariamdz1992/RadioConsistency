#!/bin/bash
#SBATCH --job-name=test_complete_integration
#SBATCH --account=def-hinat
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=test_complete_integration_%j.out
#SBATCH --error=test_complete_integration_%j.err

# Load necessary modules (ALL modules MUST be loaded before activating virtual env)
module load python/3.11 scipy-stack cuda/12.2 mpi4py opencv

# Activate your virtual environment (AFTER loading mpi4py module)
source ~/consistency_models_env/bin/activate

# Change to the consistency_models directory
cd /home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models

# Set environment variables for single GPU
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models:$PYTHONPATH"

echo "Starting complete conditioned consistency training integration test..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"

# Print environment info
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run the fixed complete integration test
echo "Running fixed pipeline integration test..."
python test_complete_integration.py

echo "Complete integration test completed at: $(date)"
