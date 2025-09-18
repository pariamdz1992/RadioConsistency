#!/bin/bash
#SBATCH --job-name=consistency_train_radiomapseer
#SBATCH --account=def-hinat
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G


# Load necessary modules (ALL modules MUST be loaded before activating virtual env)
module load python/3.11 scipy-stack cuda/12.2 mpi4py opencv

# Activate your virtual environment (AFTER loading mpi4py module)
source ~/consistency_models_env/bin/activate

# Change to the consistency_models directory
cd /home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models

# Set environment variables for single GPU
export CUDA_VISIBLE_DEVICES=0

# Create checkpoint directory
mkdir -p /home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models/checkpoints

# Run consistency training on single GPU (no MPI) - with persistent checkpoints
OPENAI_LOGDIR=/home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models/checkpoints python scripts/cm_train.py \
  --training_mode consistency_training \
  --target_ema_mode adaptive --start_ema 0.95 \
  --scale_mode progressive --start_scales 2 --end_scales 150 \
  --total_training_steps 500000 \
  --loss_norm lpips --lr_anneal_steps 0 \
  --attention_resolutions 32,16,8 \
  --class_cond False --use_scale_shift_norm False --dropout 0.0 \
  --ema_rate 0.9999,0.99994,0.9999432189950708 \
  --global_batch_size 16 \
  --image_size 256 --lr 0.0001 \
  --num_channels 128 --num_head_channels 64 --num_res_blocks 2 \
  --resblock_updown True --schedule_sampler uniform \
  --use_fp16 True --weight_decay 0.0 --weight_schedule uniform \
  --data_dir /home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models/datasets

echo "Consistency training completed!"
