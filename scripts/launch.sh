#!/usr/bin/env bash
# RadioConsistency — Unconditional training only
# Data naming expected in datasets/unconditional: gain_XXXX_YY.png
#   - XXXX: 4-digit building layout ID
#   - YY  : transmitter index (0–79)

set -euo pipefail

# repo root
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Path to datasets (change if needed, or pass as 1st arg)
DATA_DIR="${1:-/home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models/datasets}"

echo "=== Unconditional training ==="
echo "DATA_DIR: ${DATA_DIR}"
echo

# ---- run training (unconditional) ----
python scripts/cm_train.py \
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
  --data_dir "${DATA_DIR}"

# If you ever want to use the alternative module-style entrypoint you mentioned,
# keep this as a reference (commented):
# python -m orc.diffusion.scripts.train_imagenet_edm \
#   --attention_resolutions 32,16,8 --class_cond False --dropout 0.1 \
#   --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 256 \
#   --image_size 256 --lr 0.0001 --num_channels 256 --num_head_channels 64 \
#   --num_res_blocks 2 --resblock_updown True --schedule_sampler lognormal \
#   --use_fp16 True --use_scale_shift_norm False --weight_decay 0.0 \
#   --weight_schedule karras --data_dir "${DATA_DIR}"
