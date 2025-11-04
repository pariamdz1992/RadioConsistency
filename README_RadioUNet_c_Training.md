# RadioUNet_c Consistency Models Training Guide - DPM Scenarios

This README provides instructions for training consistency models with RadioUNet_c dataset using DPM simulation, with and without cars variants.

## Overview

The RadioUNet_c dataset is designed for radio map prediction using accurate buildings and no measurements. This guide focuses specifically on DPM (Dominant Path Model) simulation scenarios for consistency models training.

## Dataset Configuration

### Input Channels
The RadioUNet_c dataset provides 3-channel conditioning inputs:
- **Channel 0**: Building map (normalized buildings layout)
- **Channel 1**: Transmitter locations (antenna positions)  
- **Channel 2**: Cars channel (cars layout or zeros if cars not included)

### Target Output
- **Single channel**: Radio gain map (normalized radio propagation prediction)

## Supported Scenarios

### Scenario 1: Basic DPM (No Cars)
```bash
--dataset_type RadioUNet_c
--simulation DPM
--carsSimul no
--carsInput no
--cityMap complete
--cond_in_dim 3
```
- Cars are **not included** in the simulation
- Cars channel contains **zeros**
- Faster training, simpler scenario

### Scenario 2: DPM with Cars in Simulation
```bash
--dataset_type RadioUNet_c
--simulation DPM
--carsSimul yes
--carsInput no
--cityMap complete
--cond_in_dim 3
```
- Cars **affect the radio propagation** simulation
- Cars channel still contains **zeros** in conditioning
- More realistic simulation but same input complexity

### Scenario 3: DPM with Cars in Input
```bash
--dataset_type RadioUNet_c
--simulation DPM
--carsSimul no
--carsInput yes
--cityMap complete
--cond_in_dim 3
```
- Cars are **not in simulation** but provided as **conditioning input**
- Cars channel contains **actual car positions**
- Model learns to account for cars during inference

### Scenario 4: DPM with Cars in Both
```bash
--dataset_type RadioUNet_c
--simulation DPM
--carsSimul yes
--carsInput yes
--cityMap complete
--cond_in_dim 3
```
- Cars affect **both simulation and conditioning**
- Most complete and realistic scenario
- Cars channel contains **actual car positions**

## Job Script Template

### Complete SLURM Script: `run_radiounet_c_dpm.sh`

```bash
#!/bin/bash
#SBATCH --job-name=cm_radiounet_c_dpm
#SBATCH --account=def-hinat
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=cm_radiounet_c_dpm_%j.out
#SBATCH --error=cm_radiounet_c_dpm_%j.err

# Load necessary modules
module load python/3.11 scipy-stack cuda/12.2 mpi4py opencv

# Activate your virtual environment
source ~/consistency_models_env/bin/activate

# Change to consistency_models directory
cd /home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/pariamdz/projects/def-hinat/pariamdz/RadioDiff/consistency_models:$PYTHONPATH"

echo "Starting RadioUNet_c DPM consistency training..."
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
    --carsSimul no \
    --carsInput no \
    --cityMap complete \
    --use_fp16 False \
    --weight_decay 0.01 \
    --schedule_sampler uniform \
    --use_image_conditioning True

echo "Training completed at: $(date)"
```

## Ready-to-Use Job Scripts

### Script 1: Basic DPM (No Cars)
**File**: `run_dpm_basic.sh`

```bash
# Key parameters to modify:
--dataset_type RadioUNet_c \
--simulation DPM \
--carsSimul no \
--carsInput no \
--cityMap complete \
--cond_in_dim 3 \
--use_image_conditioning True
```

### Script 2: DPM with Cars in Simulation
**File**: `run_dpm_cars_sim.sh`

```bash
# Key parameters to modify:
--dataset_type RadioUNet_c \
--simulation DPM \
--carsSimul yes \
--carsInput no \
--cityMap complete \
--cond_in_dim 3 \
--use_image_conditioning True
```

### Script 3: DPM with Cars in Input
**File**: `run_dpm_cars_input.sh`

```bash
# Key parameters to modify:
--dataset_type RadioUNet_c \
--simulation DPM \
--carsSimul no \
--carsInput yes \
--cityMap complete \
--cond_in_dim 3 \
--use_image_conditioning True
```

### Script 4: DPM with Cars in Both
**File**: `run_dpm_cars_both.sh`

```bash
# Key parameters to modify:
--dataset_type RadioUNet_c \
--simulation DPM \
--carsSimul yes \
--carsInput yes \
--cityMap complete \
--cond_in_dim 3 \
--use_image_conditioning True
```

## Parameter Reference

### Core Dataset Parameters
- `--dataset_type RadioUNet_c`: Use RadioUNet_c dataset variant
- `--simulation DPM`: Use Dominant Path Model simulation
- `--carsSimul`: Include cars in simulation (`yes` or `no`)
- `--carsInput`: Include cars in conditioning input (`yes` or `no`) 
- `--cityMap complete`: Use complete city maps (recommended)
- `--data_dir`: Path to RadioMapSeer dataset directory

### Model Configuration
- `--cond_in_dim 3`: Always 3 for RadioUNet_c (buildings + transmitters + cars/zeros)
- `--in_channels 1`: Single channel radio gain map input
- `--out_channels 1`: Single channel radio gain map output
- `--use_conditioned_model True`: Enable conditioning
- `--use_image_conditioning True`: Use rich image conditioning
- `--cond_net swin`: Swin transformer for conditioning network

### Training Parameters
- `--image_size 64`: Resolution (can be 64, 128, or 256)
- `--batch_size 8`: Batch size (adjust based on GPU memory)
- `--lr 0.0001`: Learning rate
- `--total_training_steps 100000`: Total training iterations
- `--save_interval 5000`: Checkpoint saving frequency
- `--log_interval 100`: Logging frequency

## Quick Start Guide

1. **Choose your scenario** from the 4 DPM variants above
2. **Copy the base script** and modify the car parameters:
   ```bash
   cp run_radiounet_c_dmp.sh run_dpm_cars_input.sh
   ```
3. **Edit the parameters** in your new script:
   ```bash
   # Change these lines for cars in input scenario:
   --carsSimul no \
   --carsInput yes \
   ```
4. **Update job name** to match your scenario:
   ```bash
   #SBATCH --job-name=cm_dpm_cars_input
   ```
5. **Submit the job**:
   ```bash
   sbatch run_dpm_cars_input.sh
   ```

## Expected Dataset Structure

Your dataset directory should contain:
```
data/
├── gain/
│   ├── DPM/              # Basic DPM simulation files
│   └── carsDPM/          # DPM simulation with cars
├── png/
│   ├── buildings_complete/   # Building layout images
│   ├── antennas/            # Transmitter position images  
│   └── cars/                # Car position images
└── ...
```

## Monitoring Training

### Check Progress
```bash
# View real-time output
tail -f cm_radiounet_c_dpm_<job_id>.out

# Check for errors  
tail -f cm_radiounet_c_dpm_<job_id>.err

# Monitor job status
squeue -u $USER
```

### Key Metrics to Monitor
- **Training loss**: Should decrease over time
- **EMA decay**: Adaptive decay scheduling
- **GPU utilization**: Should be consistently high
- **Memory usage**: Watch for OOM errors

## Troubleshooting

### Common Issues
1. **Path errors**: Ensure forward slashes `/` in Linux paths
2. **Memory errors**: Reduce `--batch_size` if CUDA OOM
3. **File not found**: Verify dataset structure matches expected layout
4. **Module errors**: Check all required modules are loaded

### Performance Tips
- Start with `--image_size 64` for faster iteration
- Use `--batch_size 8` for A100 GPU with 64x64 images
- Monitor GPU memory usage and adjust accordingly

## Expected Outputs

Each training run generates:
- **Model checkpoints**: Saved every 5000 steps in model directory
- **Training logs**: Progress metrics and loss values
- **Error logs**: Any issues or warnings during training
- **Consistency models**: Ready for radio map generation

## Next Steps

1. **Compare scenarios**: Train all 4 DPM variants to compare performance
2. **Evaluate models**: Test on validation/test sets
3. **Generate predictions**: Use trained models for radio map prediction
4. **Scale resolution**: Try higher resolutions (128, 256) for better quality

## Notes

- DPM simulation is faster than IRT2/IRT4 but less physically accurate
- Cars scenarios help model learn realistic radio propagation effects
- Image conditioning provides rich spatial context for better predictions
- All scenarios use the same 3-channel conditioning format for consistency

For questions or issues, refer to the consistency models documentation or check the error logs for specific debugging information.
