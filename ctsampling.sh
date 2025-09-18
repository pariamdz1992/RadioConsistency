#!/bin/bash
#SBATCH --job-name=sample_consistency_radiomap
#SBATCH --account=def-hinat
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load necessary modules (ALL modules MUST be loaded before activating virtual env)
module load python/3.11 scipy-stack cuda/12.2 mpi4py opencv

# Activate your virtual environment (AFTER loading mpi4py module)
source ~/consistency_models_env/bin/activate

# Change to the consistency_models directory
cd /home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models

# Set environment variables for single GPU
export CUDA_VISIBLE_DEVICES=0

# Create and set specific output directory
SAMPLE_DIR="/home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models/generated_radiomaps"
mkdir -p $SAMPLE_DIR
export OPENAI_LOGDIR=$SAMPLE_DIR

echo "Starting radiomap sampling..."
echo "Output directory: $SAMPLE_DIR"

# Run sampling - matching your exact training configuration
python scripts/image_sample.py \
  --training_mode consistency_training \
  --sampler onestep \
  --generator determ \
  --batch_size 8 \
  --num_samples 32 \
  --model_path "/home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models/checkpoints/ema_0.9999_410000.pt" \
  --class_cond False \
  --image_size 256 \
  --num_channels 128 \
  --num_head_channels 64 \
  --num_res_blocks 2 \
  --attention_resolutions 32,16,8 \
  --use_scale_shift_norm False \
  --resblock_updown True \
  --use_fp16 True \
  --weight_schedule uniform \
  --steps 1

echo "Sampling completed!"
echo "Files in output directory:"
ls -la $SAMPLE_DIR/

# Convert samples to individual PNG files immediately
cd $SAMPLE_DIR
python << 'EOF'
import numpy as np
from PIL import Image
import os
import glob

print("Converting samples to individual PNG files...")

# Find the .npz file
npz_files = glob.glob("*.npz")
if not npz_files:
    print("No .npz files found!")
    exit(1)

npz_file = npz_files[0]
print(f"Loading {npz_file}")

try:
    data = np.load(npz_file)
    images = data['arr_0']
    
    print(f"Loaded {len(images)} images")
    print(f"Image shape: {images[0].shape}")
    print(f"Value range: {images.min()} to {images.max()}")
    print(f"Data type: {images.dtype}")
    
    # Create individual images directory
    os.makedirs('individual_radiomaps', exist_ok=True)
    
    # Save each sample as PNG
    for i, img in enumerate(images):
        # Images should already be uint8 from the sampling script
        pil_img = Image.fromarray(img)
        filename = f'individual_radiomaps/generated_radiomap_{i:03d}.png'
        pil_img.save(filename)
    
    print(f"Successfully saved {len(images)} radiomap images to individual_radiomaps/")
    
except Exception as e:
    print(f"Error during conversion: {e}")
    import traceback
    traceback.print_exc()
EOF

echo "Conversion completed!"
echo "Individual radiomap images saved in: $SAMPLE_DIR/individual_radiomaps/"
ls -la individual_radiomaps/ | head -10

echo "Radiomap sampling job finished!"
