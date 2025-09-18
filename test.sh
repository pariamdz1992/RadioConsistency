#!/bin/bash
#SBATCH --job-name=test_fp32_only
#SBATCH --account=def-hinat
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load python/3.11 scipy-stack cuda/12.2 mpi4py opencv
source ~/consistency_models_env/bin/activate
cd /home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models
export CUDA_VISIBLE_DEVICES=0

python -c "
import torch
from cm.script_util_radio import create_model_and_diffusion, model_and_diffusion_defaults

print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name()}')

# Use your defaults (which should have use_fp16=False)
defaults = model_and_diffusion_defaults()
print(f'Using fp16: {defaults[\"use_fp16\"]}')

model, diffusion = create_model_and_diffusion(**defaults)
model = model.cuda()

# Test forward pass
x = torch.randn(2, 1, 256, 256, device='cuda')
cond = torch.randn(2, 2, 256, 256, device='cuda')
timesteps = torch.randn(2, device='cuda')

with torch.no_grad():
    output = model(x, timesteps, cond=cond)
    print(f'Success! Output shape: {output.shape}')
    print(f'Output device: {output.device}')
    print(f'Output dtype: {output.dtype}')

print('Model test completed successfully!')
"
