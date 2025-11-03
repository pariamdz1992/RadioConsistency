#!/bin/bash
#SBATCH --job-name=cm_train_conditioned
#SBATCH --account=def-hinat
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=cm_train_conditioned_%j.out
#SBATCH --error=cm_train_conditioned_%j.err

# Load necessary modules
module load python/3.11 scipy-stack cuda/12.2 mpi4py opencv

# Activate your virtual environment  
source ~/consistency_models_env/bin/activate

# Change to consistency_models directory
cd /home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models_backup_20251103_104500

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models:$PYTHONPATH"

echo "Starting conditioned consistency training..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME" 
echo "Date: $(date)"
echo "GPU: $(nvidia-smi -L)"

# Run conditioned consistency training
python scripts/cm_train_conditioned.py \
    --data_dir /home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/data/ \
    --image_size 64 \
    --batch_size 8 \
    --lr 0.0001 \
    --total_training_steps 100000 \
    --save_interval 5000 \
    --log_interval 100 \
    --training_mode consistency_training \
    --target_ema_mode adaptive \
    --scale_mode progressive \
    --start_ema 0.95 \
    --start_scales 2 \
    --end_scales 150 \
    --use_conditioned_model True \
    --cond_net swin \
    --cond_in_dim 3 \
    --fix_bb True \
    --in_channels 1 \
    --out_channels 1 \
    --dataset_type RadioUNet_c \
    --simulation DPM \
    --use_fp16 False \
    --weight_decay 0.01 \
    --schedule_sampler uniform

echo "Training completed at: $(date)"
